"""Click sampling and occlusion-aware object masks."""

from __future__ import annotations

import numpy as np
import torch


def sample_click(
    alpha: torch.Tensor,
    object_id_map: torch.Tensor,
    alpha_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int]:
    """Sample foreground pixel (x, y) and return clicked object_id."""
    if rng is None:
        rng = np.random.default_rng()

    fg = (alpha > alpha_threshold) & (object_id_map >= 0)
    ys, xs = torch.where(fg)
    if xs.numel() == 0:
        raise ValueError("No valid foreground pixels for click sampling")

    pick = int(rng.integers(0, xs.numel()))
    x = int(xs[pick].item())
    y = int(ys[pick].item())
    object_id = int(object_id_map[y, x].item())
    return x, y, object_id


def object_mask(object_id_map: torch.Tensor, clicked_object_id: int) -> torch.Tensor:
    """Visible-only object mask: white where dominant object matches, else black."""
    return (object_id_map == clicked_object_id).to(torch.uint8) * 255
