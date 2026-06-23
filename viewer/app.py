#!/usr/bin/env python3
"""Browser viewer for splat-proj generated datasets."""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from dataset_index import DatasetIndex

VIEWER_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(VIEWER_DIR / "templates"),
    static_folder=str(VIEWER_DIR / "static"),
)
index: DatasetIndex | None = None


def _require_index() -> DatasetIndex:
    if index is None:
        abort(500, description="Dataset index not initialized")
    return index


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


@app.route("/media/<path:rel_path>")
def media(rel_path: str):
    ds = _require_index()
    try:
        path = ds.resolve_media_path(rel_path)
    except (ValueError, FileNotFoundError):
        abort(404)
    mime, _ = mimetypes.guess_type(path.name)
    return send_file(path, mimetype=mime or "application/octet-stream", conditional=True)


def create_app(dataset_dir: Path) -> Flask:
    global index
    ds = DatasetIndex(dataset_dir)
    ds.build(use_cache=True)
    index = ds
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse splat-proj dataset in the browser")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/run2"),
        help="Path to generated dataset directory (contains annotations.jsonl)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    if not (dataset_dir / "annotations.jsonl").is_file():
        raise SystemExit(f"Not a dataset directory (missing annotations.jsonl): {dataset_dir}")

    create_app(dataset_dir)
    print(f"Dataset viewer: http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_dir} ({index.count if index else 0} samples)")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
