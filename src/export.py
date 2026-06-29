"""Write PNG images and JSONL annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


@dataclass
class SampleRecord:
    id: str
    image: str
    mask: str
    point: list[int]
    object_id: int
    num_objects: int
    background: dict[str, Any]
    camera: dict[str, Any]
    objects: list[dict[str, Any]]
    augmentation: dict[str, Any] | None = None


def get_last_sample_index(output_dir: Path) -> int:
    """Return numeric index of the last sample in ``output_dir`` (0 if none)."""
    jsonl = output_dir / "annotations.jsonl"
    if jsonl.is_file():
        last_id: str | None = None
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                last_id = str(record["id"])
        if last_id is not None:
            return int(last_id)

    images = output_dir / "images"
    if images.is_dir():
        max_idx = 0
        for path in images.glob("*.png"):
            try:
                max_idx = max(max_idx, int(path.stem))
            except ValueError:
                continue
        if max_idx > 0:
            return max_idx

    return 0


def save_config_snapshot(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _tensor_to_png(path: Path, tensor: torch.Tensor, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().numpy()
    if mode == "L":
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        rgb = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def export_sample(
    output_dir: Path,
    sample_id: str,
    rgb: torch.Tensor,
    mask: torch.Tensor,
    record: SampleRecord,
) -> None:
    """Write image, mask PNGs and append JSONL record."""
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    image_rel = f"images/{sample_id}.png"
    mask_rel = f"masks/{sample_id}.png"

    _tensor_to_png(images_dir / f"{sample_id}.png", rgb, mode="RGB")
    _tensor_to_png(masks_dir / f"{sample_id}.png", mask, mode="L")

    record.image = image_rel
    record.mask = mask_rel

    jsonl_path = output_dir / "annotations.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
