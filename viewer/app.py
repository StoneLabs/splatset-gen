#!/usr/bin/env python3
"""Browser viewer for splat-proj generated datasets."""

from __future__ import annotations

import argparse
import io
import mimetypes
import sys
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from PIL import Image

from dataset_index import DatasetIndex, find_annotations_path

VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parent
TRAIN_DIR = VIEWER_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from config import load_training_config  # noqa: E402
import config as train_config  # noqa: E402
from inference import ModelRunner  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(VIEWER_DIR / "templates"),
    static_folder=str(VIEWER_DIR / "static"),
)
index: DatasetIndex | None = None
model_runner: ModelRunner | None = None
model_checkpoint_override: Path | None = None
datasets_root: Path | None = None
dataset_catalog: list[dict] = []
selected_dataset_name: str | None = None
TRAINING_CONFIG_NOT_FOUND = "training / inference config data not found"


def _require_index() -> DatasetIndex:
    if index is None:
        abort(500, description="Dataset index not initialized")
    return index


def resolve_datasets_root(raw: str) -> Path:
    """Resolve a directory that contains one or more dataset subfolders."""
    text = raw.strip()
    if not text:
        raise ValueError("datasets_dir is required")

    path = Path(text).expanduser()
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = None
        for base in (Path.cwd(), PROJECT_ROOT):
            probe = (base / path).resolve()
            if probe.is_dir():
                candidate = probe
                break
        if candidate is None:
            candidate = (Path.cwd() / path).resolve()

    if not candidate.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {candidate}")
    return candidate


def count_annotation_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def discover_datasets(root: Path) -> list[dict]:
    """List immediate child folders that contain annotation files."""
    items: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            ann = find_annotations_path(entry)
        except FileNotFoundError:
            continue
        items.append(
            {
                "name": entry.name,
                "path": str(entry.resolve()),
                "annotations_file": ann.name,
                "count": count_annotation_lines(ann),
            }
        )
    return items


def dataset_dir_for_name(name: str) -> Path:
    if datasets_root is None:
        abort(500, description="Datasets root not initialized")

    root = datasets_root.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root:
        abort(400, description="Invalid dataset name")
    find_annotations_path(candidate)
    return candidate


def load_dataset(dataset_dir: Path, *, use_cache: bool = True) -> DatasetIndex:
    global index
    ds = DatasetIndex(dataset_dir)
    ds.build(use_cache=use_cache)
    index = ds
    return ds


def refresh_dataset_catalog() -> list[dict]:
    global dataset_catalog
    if datasets_root is None:
        abort(500, description="Datasets root not initialized")
    dataset_catalog = discover_datasets(datasets_root)
    return dataset_catalog


def reload_dataset(*, name: str | None = None) -> DatasetIndex:
    global selected_dataset_name
    refresh_dataset_catalog()
    target = name or selected_dataset_name
    names = {item["name"] for item in dataset_catalog}
    if target not in names:
        if not dataset_catalog:
            abort(400, description="No datasets found")
        target = dataset_catalog[0]["name"]
    selected_dataset_name = target
    return load_dataset(dataset_dir_for_name(target), use_cache=False)


def select_dataset(name: str) -> DatasetIndex:
    global selected_dataset_name
    if not any(item["name"] == name for item in dataset_catalog):
        abort(400, description=f"Unknown dataset: {name}")
    selected_dataset_name = name
    return load_dataset(dataset_dir_for_name(name))


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.config.yaml")


def _active_checkpoint_path() -> Path | None:
    if model_runner is not None:
        return Path(model_runner.meta["checkpoint"])
    return _resolve_checkpoint_path()


def _training_config_meta() -> dict:
    ckpt = _active_checkpoint_path()
    if ckpt is None:
        return {"path": None, "yaml": TRAINING_CONFIG_NOT_FOUND, "found": False}

    config_path = _checkpoint_config_path(ckpt)
    if config_path.is_file():
        return {
            "path": str(config_path),
            "yaml": config_path.read_text(encoding="utf-8"),
            "found": True,
        }
    return {
        "path": str(config_path),
        "yaml": TRAINING_CONFIG_NOT_FOUND,
        "found": False,
    }


def _dataset_meta(ds: DatasetIndex) -> dict:
    return {
        "dataset_dir": str(ds.dataset_dir),
        "annotations_file": ds.annotations_path.name,
        "count": ds.count,
        "has_config": ds.config_path.is_file(),
        "model": _model_meta(),
    }


def _viewer_meta(ds: DatasetIndex) -> dict:
    return {
        "datasets_root": str(datasets_root) if datasets_root else None,
        "datasets": dataset_catalog,
        "selected": selected_dataset_name,
        "training_config": _training_config_meta(),
        **_dataset_meta(ds),
    }


def _resolve_checkpoint_path() -> Path | None:
    if model_checkpoint_override is not None:
        return model_checkpoint_override
    ckpt = (
        Path(train_config.cfg.INFERENCE_CHECKPOINT)
        if train_config.cfg.INFERENCE_CHECKPOINT
        else None
    )
    if ckpt is None:
        default_ckpt = Path(train_config.cfg.CHECKPOINT_DIR) / train_config.cfg.BEST_MODEL_NAME
        if default_ckpt.is_file():
            ckpt = default_ckpt
    return ckpt


def _apply_training_config_for_checkpoint(checkpoint_path: Path) -> None:
    config_path = _checkpoint_config_path(checkpoint_path)
    if config_path.is_file():
        train_config.cfg = load_training_config(config_path)


def load_model_runner() -> None:
    global model_runner
    model_runner = None
    ckpt = _resolve_checkpoint_path()
    if ckpt is None:
        return
    _apply_training_config_for_checkpoint(ckpt)
    try:
        model_runner = ModelRunner(ckpt)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")


def reload_model() -> dict:
    load_model_runner()
    return _model_meta()


def _model_meta() -> dict:
    if model_runner is None:
        return {
            "loaded": False,
            "checkpoint": None,
            "epoch": None,
            "device": None,
            "threshold": None,
            "metadata": None,
        }
    return {
        "loaded": True,
        "checkpoint": model_runner.meta["checkpoint"],
        "epoch": model_runner.meta["epoch"],
        "device": model_runner.meta["device"],
        "threshold": model_runner.mask_threshold,
        "metadata": model_runner.meta,
    }


def _compute_alpha_metrics(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Binary + soft alpha F1, matching train._batch_metrics_cpu on one sample."""
    pred = pred_u8.astype(np.float32) / 255.0
    gt = np.clip(gt_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    eps = 1e-6
    smooth = 1.0

    p_bin = pred > threshold
    t_bin = gt > threshold
    tp = float(np.logical_and(p_bin, t_bin).sum())
    fp = float(np.logical_and(p_bin, ~t_bin).sum())
    fn = float(np.logical_and(~p_bin, t_bin).sum())
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    bin_f1 = 2 * prec * rec / (prec + rec)

    inter = float((pred * gt).sum())
    alpha_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    soft_f1 = (2 * inter + eps) / (alpha_sum + gt_sum + eps)

    return {"bin_f1": bin_f1, "soft_f1": soft_f1}


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    ds = _require_index()
    return jsonify(_viewer_meta(ds))


@app.route("/api/dataset/select", methods=["POST"])
def api_dataset_select():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name") or request.args.get("name")
    if not name:
        abort(400, description="name is required")

    ds = select_dataset(str(name))
    return jsonify(_viewer_meta(ds))


@app.route("/api/dataset/reload", methods=["POST"])
def api_dataset_reload():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name") or selected_dataset_name
    ds = reload_dataset(name=str(name) if name else None)
    return jsonify(_viewer_meta(ds))


@app.route("/api/model/reload", methods=["POST"])
def api_model_reload():
    ds = _require_index()
    reload_model()
    return jsonify(_viewer_meta(ds))


@app.route("/api/training-config")
def api_training_config():
    return jsonify(_training_config_meta())


@app.route("/api/samples")
def api_samples():
    ds = _require_index()
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 100, type=int)
    return jsonify(
        {
            "offset": max(0, offset),
            "limit": limit,
            "total": ds.count,
            "items": ds.page_ids(offset, limit),
        }
    )


@app.route("/api/sample/<sample_id>")
def api_sample_by_id(sample_id: str):
    ds = _require_index()
    try:
        record = ds.read_record_by_id(sample_id)
        sample_index = ds.resolve_index(sample_id)
    except KeyError:
        abort(404, description=f"Unknown sample id: {sample_id}")
    return jsonify({"index": sample_index, "record": record})


@app.route("/api/sample/index/<int:sample_index>")
def api_sample_by_index(sample_index: int):
    ds = _require_index()
    try:
        record = ds.read_record(sample_index)
    except IndexError:
        abort(404, description=f"Sample index out of range: {sample_index}")
    return jsonify({"index": sample_index, "record": record})


@app.route("/api/config")
def api_config():
    ds = _require_index()
    return jsonify({"yaml": ds.read_config_text()})


@app.route("/api/predict/index/<int:sample_index>")
def api_predict(sample_index: int):
    if model_runner is None:
        abort(503, description="No model loaded. Pass --model with a checkpoint (.pth).")

    ds = _require_index()
    try:
        record = ds.read_record(sample_index)
    except IndexError:
        abort(404, description=f"Sample index out of range: {sample_index}")

    image_path = ds.resolve_media_path(record["image"])
    gt_path = ds.resolve_media_path(record["mask"])
    output_format = request.args.get("format", "alpha")
    visualization = request.args.get("visualization", "raw")
    background = request.args.get("background", "transparent")
    for name, value, allowed in (
        ("format", output_format, {"alpha", "binary"}),
        ("visualization", visualization, {"raw", "compare"}),
        ("background", background, {"transparent", "black"}),
    ):
        if value not in allowed:
            abort(400, description=f"{name} must be one of: {', '.join(sorted(allowed))}")

    try:
        pred = model_runner.predict_alpha(image_path, record["point"])
        gt = np.array(Image.open(gt_path).convert("L"))
        metrics = _compute_alpha_metrics(pred, gt, model_runner.mask_threshold)
        encoded, pil_mode = model_runner.encode_prediction_png(
            pred,
            output_format=output_format,
            visualization=visualization,
            background=background,
            gt_u8=gt if visualization == "compare" else None,
            threshold=model_runner.mask_threshold,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    except Exception as exc:
        print(f"Prediction error for sample {sample_index}: {exc}")
        return jsonify({"error": str(exc)}), 500

    buf = io.BytesIO()
    Image.fromarray(encoded, mode=pil_mode).save(buf, format="PNG")
    buf.seek(0)
    response = send_file(buf, mimetype="image/png", download_name=f"{record['id']}_ai.png")
    response.headers["X-AI-Bin-F1"] = f"{metrics['bin_f1']:.4f}"
    response.headers["X-AI-Soft-F1"] = f"{metrics['soft_f1']:.4f}"
    response.headers["X-AI-Threshold"] = str(model_runner.mask_threshold)
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({"error": exc.description or exc.name}), exc.code
    return exc


@app.route("/media/<path:rel_path>")
def media(rel_path: str):
    ds = _require_index()
    try:
        path = ds.resolve_media_path(rel_path)
    except (ValueError, FileNotFoundError):
        abort(404)
    mime, _ = mimetypes.guess_type(path.name)
    return send_file(path, mimetype=mime or "application/octet-stream", conditional=True)


def create_app(
    datasets_dir: Path,
    *,
    initial: str | None = None,
    checkpoint: Path | None = None,
) -> Flask:
    global index, model_runner, model_checkpoint_override, datasets_root, dataset_catalog, selected_dataset_name

    root = resolve_datasets_root(str(datasets_dir))
    datasets_root = root
    dataset_catalog = discover_datasets(root)
    if not dataset_catalog:
        raise SystemExit(
            f"No datasets found in {root} "
            "(expected subfolders with annotations.jsonl or annotations_processed.jsonl)"
        )

    names = {item["name"] for item in dataset_catalog}
    if initial and initial in names:
        start_name = initial
    else:
        start_name = dataset_catalog[0]["name"]

    select_dataset(start_name)

    model_checkpoint_override = checkpoint.resolve() if checkpoint else None
    load_model_runner()
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse splat-proj dataset in the browser")
    parser.add_argument(
        "datasets_dir",
        nargs="?",
        type=Path,
        default=Path("outputs"),
        help="Directory containing dataset folders (each with annotations.jsonl)",
    )
    parser.add_argument(
        "--initial",
        default=None,
        metavar="NAME",
        help="Initial dataset subfolder name (default: first alphabetically)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model checkpoint (.pth). Defaults to train/checkpoints/best_by_val_loss.pth if present",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        root = resolve_datasets_root(str(args.datasets_dir))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint = args.model.resolve() if args.model else None
    create_app(root, initial=args.initial, checkpoint=checkpoint)

    print(f"Dataset viewer: http://{args.host}:{args.port}")
    print(f"Datasets root: {root} ({len(dataset_catalog)} runs)")
    if selected_dataset_name and index:
        print(f"Active: {selected_dataset_name} ({index.count} samples)")
    if model_runner is not None:
        print(f"Model: {model_runner.meta['checkpoint']} (epoch {model_runner.meta['epoch']})")
    else:
        print("Model: not loaded")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
