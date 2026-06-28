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

from dataset_index import DatasetIndex

VIEWER_DIR = Path(__file__).resolve().parent
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


def _require_index() -> DatasetIndex:
    if index is None:
        abort(500, description="Dataset index not initialized")
    return index


def _model_meta() -> dict:
    if model_runner is None:
        return {
            "loaded": False,
            "checkpoint": None,
            "epoch": None,
            "device": None,
            "threshold": None,
        }
    return {
        "loaded": True,
        "checkpoint": model_runner.meta["checkpoint"],
        "epoch": model_runner.meta["epoch"],
        "device": model_runner.meta["device"],
        "threshold": model_runner.mask_threshold,
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
    return jsonify(
        {
            "dataset_dir": str(ds.dataset_dir),
            "count": ds.count,
            "has_config": ds.config_path.is_file(),
            "model": _model_meta(),
        }
    )


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
        abort(503, description="No model loaded. Pass --model or set inference.checkpoint in training_config.yaml")

    ds = _require_index()
    try:
        record = ds.read_record(sample_index)
    except IndexError:
        abort(404, description=f"Sample index out of range: {sample_index}")

    image_path = ds.resolve_media_path(record["image"])
    gt_path = ds.resolve_media_path(record["mask"])
    try:
        pred = model_runner.predict_alpha(image_path, record["point"])
        gt = np.array(Image.open(gt_path).convert("L"))
        metrics = _compute_alpha_metrics(pred, gt, model_runner.mask_threshold)
    except Exception as exc:
        print(f"Prediction error for sample {sample_index}: {exc}")
        return jsonify({"error": str(exc)}), 500

    buf = io.BytesIO()
    Image.fromarray(pred, mode="L").save(buf, format="PNG")
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


def create_app(dataset_dir: Path, checkpoint: Path | None = None) -> Flask:
    global index, model_runner
    ds = DatasetIndex(dataset_dir)
    ds.build(use_cache=True)
    index = ds

    model_runner = None
    ckpt = checkpoint or (
        Path(train_config.cfg.INFERENCE_CHECKPOINT)
        if train_config.cfg.INFERENCE_CHECKPOINT
        else None
    )
    if ckpt is None:
        default_ckpt = Path(train_config.cfg.CHECKPOINT_DIR) / train_config.cfg.BEST_MODEL_NAME
        if default_ckpt.is_file():
            ckpt = default_ckpt
    if ckpt is not None:
        try:
            model_runner = ModelRunner(ckpt)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse splat-proj dataset in the browser")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/run2"),
        help="Path to generated dataset directory (contains annotations.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional model checkpoint (.pth). Defaults to inference.checkpoint in training_config.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=TRAIN_DIR / "training_config.yaml",
        help="Path to train/training_config.yaml",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train_config.cfg = load_training_config(args.training_config)

    dataset_dir = args.dataset.resolve()
    if not (dataset_dir / "annotations.jsonl").is_file():
        raise SystemExit(f"Not a dataset directory (missing annotations.jsonl): {dataset_dir}")

    checkpoint = args.model.resolve() if args.model else None
    create_app(dataset_dir, checkpoint=checkpoint)

    print(f"Dataset viewer: http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_dir} ({index.count if index else 0} samples)")
    if model_runner is not None:
        print(f"Model: {model_runner.meta['checkpoint']} (epoch {model_runner.meta['epoch']})")
    else:
        print("Model: not loaded")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
