"""Background compositing after splat pass (solid color, image, or random pixels)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class BackgroundSpec:
    mode: str
    solid_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    image_dir: Path | None = None
    resize_mode: str = "crop"


def background_from_config(
    config: dict[str, Any],
    base_dir: Path | None = None,
) -> BackgroundSpec:
    """Build ``BackgroundSpec`` from YAML config (paths relative to ``base_dir``)."""
    base = base_dir or Path.cwd()
    bg = config.get("background", {})
    color = bg.get("solid_color", [0.1, 0.1, 0.1])
    image_dir_raw = bg.get("image_dir")
    image_dir: Path | None = None
    if image_dir_raw:
        raw = Path(image_dir_raw)
        image_dir = raw if raw.is_absolute() else (base / raw).resolve()
    return BackgroundSpec(
        mode=bg.get("mode", "solid"),
        solid_color=(float(color[0]), float(color[1]), float(color[2])),
        image_dir=image_dir,
        resize_mode=bg.get("resize_mode", "crop"),
    )


def list_background_images(image_dir: Path) -> list[Path]:
    """Return sorted image paths under ``image_dir``."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Background image directory not found: {image_dir}")
    paths = [
        p
        for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No background images in {image_dir}")
    return paths


def load_background_image(
    path: Path,
    width: int,
    height: int,
    resize_mode: str,
    solid_color: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> torch.Tensor:
    """Load and resize a background image to ``[height, width, 3]`` float RGB in [0, 1]."""
    if resize_mode not in {"crop", "letterbox", "stretch"}:
        raise ValueError(f"Unknown resize_mode {resize_mode!r}; use crop, letterbox, or stretch")

    img = Image.open(path).convert("RGB")
    if resize_mode == "stretch":
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
    else:
        if resize_mode == "crop":
            scale = max(width / img.width, height / img.height)
        else:
            scale = min(width / img.width, height / img.height)
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if resize_mode == "crop":
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            resized = resized.crop((left, top, left + width, top + height))
        else:
            pad = tuple(int(round(c * 255.0)) for c in solid_color)
            canvas = Image.new("RGB", (width, height), pad)
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            resized = canvas

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def sample_background_layer(
    background: BackgroundSpec,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return background RGB tensor and metadata for JSONL."""
    if background.mode == "solid":
        color = background.solid_color
        bg = torch.tensor(color, dtype=torch.float32).view(1, 1, 3).expand(height, width, 3).clone()
        return bg, {"mode": "solid", "color": list(color)}

    if background.mode == "random_pixels":
        arr = rng.random((height, width, 3), dtype=np.float32)
        return torch.from_numpy(arr), {"mode": "random_pixels"}

    if background.mode != "image":
        raise ValueError(f"Unknown background mode {background.mode!r}")

    if background.image_dir is None:
        raise ValueError("background.mode='image' requires background.image_dir")

    images = list_background_images(background.image_dir)
    chosen = images[int(rng.integers(0, len(images)))]
    bg = load_background_image(
        chosen,
        width,
        height,
        background.resize_mode,
        background.solid_color,
    )
    return bg, {
        "mode": "image",
        "image": chosen.name,
        "image_dir": str(background.image_dir),
        "resize_mode": background.resize_mode,
    }


def composite(
    fg_rgb: torch.Tensor,
    alpha: torch.Tensor,
    background: BackgroundSpec,
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Composite foreground splats over solid color, random pixels, or a background image.

    Masks and click sampling always use ``object_id_map`` / ``alpha`` from the
    splat pass — never post-composite RGB.
    """
    if background.mode == "solid":
        bg_rgb, meta = sample_background_layer(
            background,
            width,
            height,
            rng or np.random.default_rng(),
        )
    elif background.mode in {"image", "random_pixels"}:
        if rng is None:
            raise ValueError(
                f"background.mode={background.mode!r} requires an RNG for random background selection"
            )
        bg_rgb, meta = sample_background_layer(background, width, height, rng)
    else:
        raise ValueError(f"Unknown background mode {background.mode!r}")

    device = fg_rgb.device
    dtype = fg_rgb.dtype
    bg_rgb = bg_rgb.to(device=device, dtype=dtype)
    a = alpha.unsqueeze(-1)
    rgb = fg_rgb * a + bg_rgb * (1.0 - a)
    return rgb, meta
