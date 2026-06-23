"""Background compositing after splat pass (v1: solid color)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BackgroundSpec:
    mode: str
    solid_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    image_dir: str | None = None
    resize_mode: str = "crop"


def composite(
    fg_rgb: torch.Tensor,
    alpha: torch.Tensor,
    background: BackgroundSpec,
    width: int,
    height: int,
) -> torch.Tensor:
    """Composite foreground splats over solid color or image.

    v1 implements ``mode == \"solid\"`` only.

    v2 (planned): when ``mode == \"image\"``, load a random image from
    ``background.image_dir``, resize per ``resize_mode`` (crop | letterbox |
    stretch), then composite with the same alpha blend:

        rgb = fg_rgb * alpha + background_image * (1 - alpha)

    Masks and click sampling always use ``object_id_map`` / ``alpha`` from the
    splat pass — never post-composite RGB.
    """
    if background.mode != "solid":
        raise NotImplementedError(
            f"background mode {background.mode!r} is v2; use mode='solid'"
        )

    bg = torch.tensor(background.solid_color, dtype=fg_rgb.dtype, device=fg_rgb.device)
    bg = bg.view(1, 1, 3).expand(height, width, 3)
    a = alpha.unsqueeze(-1)
    return fg_rgb * a + bg * (1.0 - a)
