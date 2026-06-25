"""Click sampling and occlusion-aware object masks."""

from __future__ import annotations

import numpy as np
import torch

MASK_MODES = frozenset({"binary", "soft"})


def sample_click(
    alpha: torch.Tensor,
    object_id_map: torch.Tensor,
    alpha_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
    *,
    object_weights: torch.Tensor | None = None,
    weight_threshold: float = 0.05,
) -> tuple[int, int, int]:
    """Sample foreground pixel (x, y) and return clicked object_id."""
    if rng is None:
        rng = np.random.default_rng()

    fg = (alpha > alpha_threshold) & (object_id_map >= 0)
    if object_weights is not None and object_weights.numel() > 0:
        oid = object_id_map.clamp(min=0)
        contrib = object_weights.gather(-1, oid.unsqueeze(-1)).squeeze(-1)
        fg = fg & (contrib > weight_threshold)

    ys, xs = torch.where(fg)
    if xs.numel() == 0:
        raise ValueError("No valid foreground pixels for click sampling")

    pick = int(rng.integers(0, xs.numel()))
    x = int(xs[pick].item())
    y = int(ys[pick].item())
    object_id = int(object_id_map[y, x].item())
    return x, y, object_id


def object_mask(
    object_weights: torch.Tensor,
    clicked_object_id: int,
    *,
    mode: str = "binary",
    weight_threshold: float = 0.05,
) -> torch.Tensor:
    """Visible-only object mask from per-object compositing weights.

    ``binary`` (default): 0 or 255 — white where the clicked object is dominant
    and its accumulated weight exceeds ``weight_threshold``; black elsewhere.

    ``soft``: grayscale 0–255 proportional to accumulated weight on dominant
    pixels (matches semi-transparent splat edges). Pixels below
    ``weight_threshold`` are zeroed as noise. Occluded / non-dominant pixels stay 0.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown mask mode {mode!r}; use binary or soft")

    contrib = object_weights[:, :, clicked_object_id]
    dominant = object_weights.argmax(dim=-1) == clicked_object_id

    if mode == "soft":
        strength = torch.where(dominant, contrib, torch.zeros_like(contrib))
        if weight_threshold > 0.0:
            strength = torch.where(
                strength >= weight_threshold,
                strength,
                torch.zeros_like(strength),
            )
        return (strength.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    visible = (contrib > weight_threshold) & dominant
    return visible.to(torch.uint8) * 255


def click_inside_mask(mask: torch.Tensor, x: int, y: int, *, mode: str = "binary") -> bool:
    """Return True if ``(x, y)`` lies inside the exported mask."""
    value = int(mask[y, x].item())
    if mode == "binary":
        return value == 255
    return value > 0


def count_mask_pixels(mask: torch.Tensor, *, mode: str = "binary") -> int:
    """Count foreground mask pixels (255 for binary, any value > 0 for soft)."""
    if mode == "binary":
        return int((mask == 255).sum().item())
    return int((mask > 0).sum().item())
