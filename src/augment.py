"""Random lighting and camera-style post-processing for training RGB images."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

AUGMENTATION_SEED_STRIDE = 1_000_003
GAUSSIAN_SUBSET_SEED_STRIDE = 1_000_033
AUGMENTATION_RETRY_STRIDE = 97


def make_augmentation_rng(
    master_seed: int,
    sample_id: str,
    attempt: int = 0,
) -> np.random.Generator:
    """Deterministic augmentation RNG from run seed + sample id (worker-independent)."""
    return np.random.default_rng(
        master_seed
        + int(sample_id) * AUGMENTATION_SEED_STRIDE
        + int(attempt) * AUGMENTATION_RETRY_STRIDE
    )


def make_gaussian_subset_rng(master_seed: int, sample_id: str) -> np.random.Generator:
    """Worker-independent RNG for optional pre-render Gaussian subsetting."""
    return np.random.default_rng(master_seed + int(sample_id) * GAUSSIAN_SUBSET_SEED_STRIDE)


def get_augmentation_config(config: dict[str, Any]) -> dict[str, Any]:
    if "augmentation" in config:
        return config["augmentation"]
    return config.get("generation", {}).get("augmentation", {})


def maybe_gaussian_draw_fraction_range(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[float] | None:
    """Return Gaussian draw fraction range when gaussian_subset augmentation triggers."""
    aug_cfg = get_augmentation_config(config)
    if not aug_cfg.get("enabled", False):
        return None

    subset_cfg = aug_cfg.get("gaussian_subset", {})
    if not subset_cfg.get("enabled", False):
        return None
    if rng.random() >= float(subset_cfg.get("probability", 1.0)):
        return None

    frac = subset_cfg.get("fraction_range", [0.1, 1.0])
    return [float(frac[0]), float(frac[1])]


def _append_effect(applied: list[dict[str, Any]], effect: str, params: dict[str, Any]) -> None:
    applied.append({"effect": effect, **params})


def gaussian_subset_augmentation_entry(config: dict[str, Any]) -> dict[str, Any] | None:
    """Record whether gaussian_subset is enabled in config (details live on each object)."""
    aug_cfg = get_augmentation_config(config)
    if "gaussian_subset" not in aug_cfg:
        return None
    return {
        "effect": "gaussian_subset",
        "enabled": bool(aug_cfg.get("gaussian_subset", {}).get("enabled", False)),
    }


def init_augmentation_record(config: dict[str, Any]) -> dict[str, Any] | None:
    """Start augmentation metadata with any pre-render effect flags."""
    if not get_augmentation_config(config).get("enabled", False):
        return None

    applied: list[dict[str, Any]] = []
    subset = gaussian_subset_augmentation_entry(config)
    if subset is not None:
        applied.append(subset)
    return {"enabled": True, "applied": applied}


def merge_post_augmentation(
    record: dict[str, Any] | None,
    post_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Append post-render augmentation effects in application order."""
    if not post_meta.get("enabled"):
        return record
    if record is None:
        return post_meta
    record["applied"].extend(post_meta.get("applied", []))
    return record


def append_lines_augmentation(
    record: dict[str, Any] | None,
    lines_meta: dict[str, Any],
) -> dict[str, Any] | None:
    if record is None:
        record = {"enabled": True, "applied": []}
    _append_effect(record["applied"], "lines", lines_meta)
    return record


def finalize_augmentation_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None or not record.get("applied"):
        return None
    return record


def format_augmentation_log(meta: dict[str, Any]) -> str:
    """One-line summary of applied augmentation effects for verbose logging."""
    labels: list[str] = []
    for entry in meta.get("applied", []):
        effect = entry.get("effect", "?")
        if effect == "gaussian_subset":
            state = "on" if entry.get("enabled") else "off"
            labels.append(f"gaussian_subset={state}")
        elif effect == "lines":
            labels.append(f"lines×{entry.get('count', 0)}")
        else:
            labels.append(effect)
    return " → ".join(labels)


def format_augmentation_effect_detail(entry: dict[str, Any]) -> str:
    """Compact parameter summary for one applied effect."""
    effect = entry.get("effect", "?")
    if effect == "gaussian_subset":
        return "enabled" if entry.get("enabled") else "disabled"
    if effect == "lines":
        return f"count={entry.get('count', 0)}"
    params = {key: value for key, value in entry.items() if key != "effect"}
    return str(params)


def _clip_rgb(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _uniform(rng: np.random.Generator, key: str, cfg: dict[str, Any], default: list[float]) -> float:
    lo, hi = cfg.get(key, default)
    return float(rng.uniform(float(lo), float(hi)))


def _int_uniform(rng: np.random.Generator, key: str, cfg: dict[str, Any], default: list[int]) -> int:
    lo, hi = cfg.get(key, default)
    return int(rng.integers(int(lo), int(hi) + 1))


def _probability(cfg: dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _maybe_apply(
    rng: np.random.Generator,
    cfg: dict[str, Any],
    fn,
    arr: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not cfg.get("enabled", False):
        return arr, None
    probability = float(cfg.get("probability", 1.0))
    if rng.random() >= probability:
        return arr, None
    return fn(arr, cfg, rng)


def apply_lighting(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    brightness = _uniform(rng, "brightness_range", cfg, [-0.2, 0.2])
    contrast = _uniform(rng, "contrast_range", cfg, [0.8, 1.2])
    gamma = _uniform(rng, "gamma_range", cfg, [0.85, 1.15])
    saturation = _uniform(rng, "saturation_range", cfg, [0.7, 1.3])
    warmth = _uniform(rng, "warmth_range", cfg, [-0.15, 0.15])

    out = (arr - 0.5) * contrast + 0.5 + brightness
    out = _clip_rgb(out) ** (1.0 / max(gamma, 1e-3))

    gray = out.mean(axis=2, keepdims=True)
    out = gray + saturation * (out - gray)

    out[..., 0] += warmth
    out[..., 2] -= warmth

    meta = {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "gamma": round(gamma, 4),
        "saturation": round(saturation, 4),
        "warmth": round(warmth, 4),
    }
    return _clip_rgb(out), meta


def apply_blur(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    motion_prob = _probability(cfg, "motion_probability", 0.5)

    if rng.random() < motion_prob:
        length = _int_uniform(rng, "motion_length_range", cfg, [3, 15])
        angle_deg = _uniform(rng, "motion_angle_range", cfg, [0.0, 180.0])
        meta = {
            "type": "motion",
            "length": length,
            "angle_deg": round(angle_deg, 2),
        }
        return apply_blur_meta(arr, meta), meta

    radius = _uniform(rng, "radius_range", cfg, [0.5, 2.5])
    meta = {"type": "gaussian", "radius": round(radius, 3)}
    return apply_blur_meta(arr, meta), meta


def apply_blur_meta(arr: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    if meta["type"] == "motion":
        if arr.ndim == 2:
            work = arr.astype(np.float32) / 255.0
            out = _motion_blur_angle(work, int(meta["length"]), float(meta["angle_deg"]))
            return np.clip(out * 255.0, 0.0, 255.0).astype(arr.dtype)
        out = _motion_blur_angle(arr, int(meta["length"]), float(meta["angle_deg"]))
        return _clip_rgb(out)

    radius = float(meta["radius"])
    if arr.ndim == 2:
        work = arr.astype(np.float32) / 255.0
        img = Image.fromarray((work * 255.0).astype(np.uint8), mode="L")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=arr.dtype)

    img = Image.fromarray((_clip_rgb(arr) * 255.0).astype(np.uint8))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _motion_blur_angle(arr: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    length = max(1, int(length))
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    pad = length
    if arr.ndim == 2:
        work = arr.astype(np.float32)
        padded = np.pad(work, ((pad, pad), (pad, pad)), mode="edge")
        out = np.zeros_like(work)
        h, w = work.shape
        for i in range(length):
            t = i - length // 2
            oy = int(round(t * dy))
            ox = int(round(t * dx))
            out += padded[pad + oy : pad + oy + h, pad + ox : pad + ox + w]
        return out / float(length)

    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    for i in range(length):
        t = i - length // 2
        oy = int(round(t * dy))
        ox = int(round(t * dx))
        out += padded[pad + oy : pad + oy + h, pad + ox : pad + ox + w]
    return out / float(length)


def _remap_bilinear(arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    src_x = np.clip(src_x, 0.0, w - 1.0)
    src_y = np.clip(src_y, 0.0, h - 1.0)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (src_x - x0)[..., np.newaxis] if arr.ndim == 3 else (src_x - x0)
    wy = (src_y - y0)[..., np.newaxis] if arr.ndim == 3 else (src_y - y0)

    if arr.ndim == 3:
        c00 = arr[y0, x0]
        c10 = arr[y0, x1]
        c01 = arr[y1, x0]
        c11 = arr[y1, x1]
        return (
            c00 * (1.0 - wx) * (1.0 - wy)
            + c10 * wx * (1.0 - wy)
            + c01 * (1.0 - wx) * wy
            + c11 * wx * wy
        ).astype(np.float32)

    c00 = arr[y0, x0]
    c10 = arr[y0, x1]
    c01 = arr[y1, x0]
    c11 = arr[y1, x1]
    return (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    ).astype(arr.dtype)


def _remap_nearest(arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    xi = np.clip(np.rint(src_x), 0, w - 1).astype(np.int32)
    yi = np.clip(np.rint(src_y), 0, h - 1).astype(np.int32)
    return arr[yi, xi]


def _affine_src_map(
    height: int,
    width: int,
    angle_deg: float,
    scale: float,
    shear: float,
    tx: float,
    ty: float,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    a = scale * (c + shear * s)
    b = scale * (-s + shear * c)
    d = scale * s
    e = scale * c
    det = a * e - b * d
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xs - cx - tx
    dy = ys - cy - ty
    src_x = inv_a * dx + inv_b * dy + cx
    src_y = inv_d * dx + inv_e * dy + cy
    return src_x, src_y


def _barrel_src_map(height: int, width: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xs - cx) / max(cx, 1.0)
    ny = (ys - cy) / max(cy, 1.0)
    r2 = nx * nx + ny * ny
    factor = 1.0 + strength * r2
    src_x = cx + (xs - cx) / factor
    src_y = cy + (ys - cy) / factor
    return src_x, src_y


def _sample_perspective_quad(
    width: int,
    height: int,
    jitter_frac: float,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    span = jitter_frac * min(width, height)
    w, h = float(width - 1), float(height - 1)
    return (
        rng.uniform(0.0, span),
        rng.uniform(0.0, span),
        w - rng.uniform(0.0, span),
        rng.uniform(0.0, span),
        w - rng.uniform(0.0, span),
        h - rng.uniform(0.0, span),
        rng.uniform(0.0, span),
        h - rng.uniform(0.0, span),
    )


def _warp_quad(arr: np.ndarray, quad: tuple[float, ...], *, nearest: bool) -> np.ndarray:
    height, width = arr.shape[:2]
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if arr.ndim == 3:
        img = Image.fromarray((_clip_rgb(arr) * 255.0).astype(np.uint8))
        out = img.transform((width, height), Image.Transform.QUAD, quad, resample)
        return np.asarray(out, dtype=np.float32) / 255.0
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    out = img.transform((width, height), Image.Transform.QUAD, quad, resample)
    return np.asarray(out, dtype=np.float32)


def _enabled_warp_types(cfg: dict[str, Any]) -> list[str]:
    types: list[str] = []
    for name in ("perspective", "barrel", "pincushion", "affine"):
        if cfg.get(name, {}).get("enabled", False):
            types.append(name)
    return types


def sample_warp_plan(
    shape: tuple[int, int],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    types = _enabled_warp_types(cfg)
    if not types:
        return None, None

    kind = types[int(rng.integers(0, len(types)))]
    height, width = shape

    if kind == "perspective":
        jitter = _uniform(rng, "corner_jitter_range", cfg["perspective"], [0.02, 0.12])
        quad = _sample_perspective_quad(width, height, jitter, rng)
        plan = {"type": "perspective", "quad": quad}
        meta = {"type": "perspective", "corner_jitter_frac": round(jitter, 4)}
        return plan, meta

    if kind == "barrel":
        strength = _uniform(rng, "strength_range", cfg["barrel"], [-0.2, -0.02])
        plan = {"type": "barrel", "strength": strength}
        meta = {"type": "barrel", "strength": round(strength, 4)}
        return plan, meta

    if kind == "pincushion":
        strength = _uniform(rng, "strength_range", cfg["pincushion"], [0.02, 0.15])
        plan = {"type": "pincushion", "strength": strength}
        meta = {"type": "pincushion", "strength": round(strength, 4)}
        return plan, meta

    affine_cfg = cfg["affine"]
    angle_deg = _uniform(rng, "rotate_deg_range", affine_cfg, [-8.0, 8.0])
    scale = _uniform(rng, "scale_range", affine_cfg, [0.95, 1.05])
    shear = _uniform(rng, "shear_range", affine_cfg, [-0.08, 0.08])
    translate_lo, translate_hi = affine_cfg.get("translate_frac_range", [-0.04, 0.04])
    tx = float(rng.uniform(float(translate_lo), float(translate_hi)) * width)
    ty = float(rng.uniform(float(translate_lo), float(translate_hi)) * height)
    plan = {
        "type": "affine",
        "angle_deg": angle_deg,
        "scale": scale,
        "shear": shear,
        "tx": tx,
        "ty": ty,
    }
    meta = {
        "type": "affine",
        "angle_deg": round(angle_deg, 2),
        "scale": round(scale, 4),
        "shear": round(shear, 4),
        "translate_px": [round(tx, 2), round(ty, 2)],
    }
    return plan, meta


def _finalize_warped_mask(mask_out: np.ndarray, mask_mode: str) -> np.ndarray:
    mask_out = np.clip(mask_out, 0.0, 255.0)
    if mask_mode == "binary":
        return (mask_out >= 127.0).astype(np.uint8) * 255
    return mask_out.astype(np.uint8)


def _warp_mask_bilinear(mask_arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, mask_mode: str) -> np.ndarray:
    warped = _remap_bilinear(mask_arr.astype(np.float32) / 255.0, src_x, src_y)
    return _finalize_warped_mask(warped * 255.0, mask_mode)


def apply_warp_plan(
    rgb_arr: np.ndarray,
    mask_arr: np.ndarray | None,
    plan: dict[str, Any],
    *,
    mask_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = rgb_arr.shape[:2]
    kind = plan["type"]

    if kind == "perspective":
        rgb_out = _warp_quad(rgb_arr, plan["quad"], nearest=False)
        mask_out = None
        if mask_arr is not None:
            mask_out = _warp_quad(mask_arr.astype(np.float32) / 255.0, plan["quad"], nearest=False)
            mask_out = _finalize_warped_mask(mask_out * 255.0, mask_mode)
        return rgb_out, mask_out

    if kind in {"barrel", "pincushion"}:
        src_x, src_y = _barrel_src_map(height, width, plan["strength"])
        rgb_out = _remap_bilinear(rgb_arr, src_x, src_y)
        mask_out = (
            _warp_mask_bilinear(mask_arr, src_x, src_y, mask_mode) if mask_arr is not None else None
        )
        return _clip_rgb(rgb_out), mask_out

    src_x, src_y = _affine_src_map(
        height,
        width,
        plan["angle_deg"],
        plan["scale"],
        plan["shear"],
        plan["tx"],
        plan["ty"],
    )
    rgb_out = _remap_bilinear(rgb_arr, src_x, src_y)
    mask_out = _warp_mask_bilinear(mask_arr, src_x, src_y, mask_mode) if mask_arr is not None else None
    return _clip_rgb(rgb_out), mask_out


def apply_tear(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    plan, meta = sample_tear_plan(arr.shape[:2], cfg, rng)
    out = apply_tear_plan(arr, plan)
    return out, meta


def sample_tear_plan(
    shape: tuple[int, int],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    h, w = shape
    num_bands = _int_uniform(rng, "num_bands_range", cfg, [2, 6])
    shift_lo, shift_hi = cfg.get("shift_range", cfg.get("max_shift_range", [-20, 20]))
    horizontal = bool(rng.random() < _probability(cfg, "horizontal_probability", 0.5))

    span = h if horizontal else w
    if num_bands > 1 and span > num_bands:
        cuts = sorted(rng.choice(span - 1, size=num_bands - 1, replace=False) + 1)
    else:
        cuts = []
    boundaries = [0, *cuts, span]
    shifts = [int(rng.integers(int(shift_lo), int(shift_hi) + 1)) for _ in range(len(boundaries) - 1)]

    plan = {
        "horizontal": horizontal,
        "boundaries": boundaries,
        "shifts": shifts,
    }
    meta = {
        "orientation": "horizontal" if horizontal else "vertical",
        "num_bands": num_bands,
        "shift_range": [int(shift_lo), int(shift_hi)],
        "shifts": shifts,
    }
    return plan, meta


def apply_tear_plan(arr: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    horizontal = bool(plan["horizontal"])
    boundaries = plan["boundaries"]
    shifts = plan["shifts"]
    out = arr.copy()

    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        shift = shifts[idx]
        if horizontal:
            band = out[start:end].copy()
            out[start:end] = np.roll(band, shift, axis=1)
        else:
            band = out[:, start:end].copy()
            out[:, start:end] = np.roll(band, shift, axis=0)

    return out


def apply_noise(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    std_lo, std_hi = cfg.get("std_range", [0.01, 0.06])
    std = float(rng.uniform(float(std_lo), float(std_hi)))
    noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
    return _clip_rgb(arr + noise), {"std": round(std, 4)}


def apply_jpeg(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    q_lo, q_hi = cfg.get("quality_range", [30, 85])
    quality = int(rng.integers(int(q_lo), int(q_hi) + 1))
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return _clip_rgb(decoded), {"quality": quality}


def apply_vignette(arr: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    strength = _uniform(rng, "strength_range", cfg, [0.2, 0.6])
    falloff_start = _uniform(rng, "falloff_start_range", cfg, [0.25, 0.45])
    h, w, _ = arr.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (y - cy) / max(cy, 1.0)
    nx = (x - cx) / max(cx, 1.0)
    dist = np.sqrt(nx * nx + ny * ny)
    mask = 1.0 - strength * np.clip(dist - falloff_start, 0.0, 1.0)
    out = arr * mask[..., np.newaxis]
    return _clip_rgb(out), {
        "strength": round(strength, 4),
        "falloff_start": round(falloff_start, 4),
    }


def apply_chromatic_aberration(
    arr: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    shift = _int_uniform(rng, "shift_range", cfg, [1, 4])
    horizontal = bool(rng.random() < _probability(cfg, "horizontal_probability", 0.5))
    out = arr.copy()
    axis = 1 if horizontal else 0
    out[..., 0] = np.roll(arr[..., 0], shift, axis=axis)
    out[..., 2] = np.roll(arr[..., 2], -shift, axis=axis)
    return out, {
        "shift_px": shift,
        "orientation": "horizontal" if horizontal else "vertical",
    }


def _bbox_line_endpoints(
    width: int,
    height: int,
    px: float,
    py: float,
    dx: float,
    dy: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    w, h = float(width - 1), float(height - 1)
    points: list[tuple[float, float]] = []
    eps = 1e-6

    if abs(dx) > eps:
        for x in (0.0, w):
            y = py + ((x - px) / dx) * dy
            if -1e-6 <= y <= h + 1e-6:
                points.append((x, float(np.clip(y, 0.0, h))))
    if abs(dy) > eps:
        for y in (0.0, h):
            x = px + ((y - py) / dy) * dx
            if -1e-6 <= x <= w + 1e-6:
                points.append((float(np.clip(x, 0.0, w)), y))

    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(abs(point[0] - kept[0]) < 0.01 and abs(point[1] - kept[1]) < 0.01 for kept in unique):
            unique.append(point)

    if len(unique) < 2:
        return (0.0, 0.0), (w, h)

    best_pair = unique[0], unique[1]
    best_dist = -1.0
    for i, p0 in enumerate(unique):
        for p1 in unique[i + 1 :]:
            dist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = p0, p1
    return best_pair


def _spanning_line_endpoints(
    width: int,
    height: int,
    angle_deg: float,
    offset_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    angle_rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    nx, ny = -dy, dx
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    px = cx + nx * offset_px
    py = cy + ny * offset_px
    return _bbox_line_endpoints(width, height, px, py, dx, dy)


def _rasterize_line(
    width: int,
    height: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_width: int,
) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.line([(x0, y0), (x1, y1)], fill=255, width=max(1, int(line_width)))
    return np.asarray(img, dtype=bool)


def _sample_line_avoiding_point(
    width: int,
    height: int,
    click_x: int,
    click_y: int,
    line_width: int,
    angle_deg: float,
    rng: np.random.Generator,
    max_attempts: int,
    offset_range: list[float] | None,
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, float] | None:
    for _ in range(max_attempts):
        offset_px = 0.0
        if offset_range is not None:
            offset_px = float(rng.uniform(float(offset_range[0]), float(offset_range[1])))
        p0, p1 = _spanning_line_endpoints(width, height, angle_deg, offset_px)
        pixels = _rasterize_line(width, height, p0[0], p0[1], p1[0], p1[1], line_width)
        if not pixels[click_y, click_x]:
            return p0, p1, pixels, offset_px
    return None


def _line_thickness_range(cfg: dict[str, Any]) -> list[int]:
    return cfg.get("thickness_range", cfg.get("width_range", [1, 4]))


def _random_line_color(cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    channel_lo, channel_hi = cfg.get("color_range", [0.05, 1.0])
    per_channel = cfg.get("color_per_channel", True)
    if per_channel:
        return rng.uniform(float(channel_lo), float(channel_hi), size=3).astype(np.float32)
    value = float(rng.uniform(float(channel_lo), float(channel_hi)))
    return np.full(3, value, dtype=np.float32)


def apply_random_lines(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    click_x: int,
    click_y: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """Draw random colored lines on RGB; erase clicked-object mask where lines overlap.

    Lines are rejected when they would cover the click pixel ``(click_x, click_y)``.
    """
    aug_cfg = get_augmentation_config(config)
    lines_cfg = aug_cfg.get("lines", {})
    if not aug_cfg.get("enabled", False) or not lines_cfg.get("enabled", False):
        return rgb, mask, None

    probability = float(lines_cfg.get("probability", 1.0))
    if rng.random() >= probability:
        return rgb, mask, None

    count_lo, count_hi = lines_cfg.get("count_range", [1, 5])
    max_attempts = int(lines_cfg.get("max_attempts_per_line", 32))
    target_count = int(rng.integers(int(count_lo), int(count_hi) + 1))

    arr = rgb.detach().cpu().numpy().astype(np.float32).copy()
    mask_arr = mask.detach().cpu().numpy().copy()
    height, width = arr.shape[:2]
    object_fg = mask_arr > 0

    if "offset_range" in lines_cfg:
        offset_range = lines_cfg.get("offset_range")
    else:
        span = max(width, height)
        offset_range = [-float(span), float(span)]

    drawn: list[dict[str, Any]] = []
    for _ in range(target_count):
        angle_deg = _uniform(rng, "angle_deg_range", lines_cfg, [0.0, 180.0])
        line_width = _int_uniform(rng, "thickness_range", lines_cfg, _line_thickness_range(lines_cfg))
        sampled = _sample_line_avoiding_point(
            width,
            height,
            click_x,
            click_y,
            line_width,
            angle_deg,
            rng,
            max_attempts,
            offset_range,
        )
        if sampled is None:
            continue

        p0, p1, line_pixels, offset_px = sampled
        color = _random_line_color(lines_cfg, rng)
        arr[line_pixels] = color
        mask_arr[np.logical_and(line_pixels, object_fg)] = 0
        drawn.append(
            {
                "start": [round(p0[0], 2), round(p0[1], 2)],
                "end": [round(p1[0], 2), round(p1[1], 2)],
                "angle_deg": round(angle_deg, 2),
                "thickness": line_width,
                "offset_px": round(offset_px, 2),
                "color": [round(float(c), 4) for c in color],
            }
        )

    if not drawn:
        return rgb, mask, None

    out_rgb = torch.from_numpy(_clip_rgb(arr)).to(device=rgb.device, dtype=rgb.dtype)
    out_mask = torch.from_numpy(mask_arr.astype(np.uint8)).to(device=mask.device, dtype=mask.dtype)
    return out_rgb, out_mask, {"count": len(drawn), "lines": drawn}


def apply_augmentation(
    rgb: torch.Tensor,
    config: dict[str, Any],
    rng: np.random.Generator,
    mask: torch.Tensor | None = None,
    *,
    mask_mode: str = "binary",
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
    """Apply optional lighting + camera post-processing to composited RGB.

    Geometric effects (tear, warp) run first on RGB and mask together. Blur uses
    the same kernel on both. Other photometric effects are RGB-only.
    """
    aug_cfg = get_augmentation_config(config)
    if not aug_cfg.get("enabled", False):
        return rgb, {}, mask

    rgb_arr = rgb.detach().cpu().numpy().astype(np.float32)
    mask_arr: np.ndarray | None = None
    if mask is not None:
        mask_arr = mask.detach().cpu().numpy().copy()

    applied: list[dict[str, Any]] = []

    tear_cfg = aug_cfg.get("tear", {})
    if tear_cfg.get("enabled", False) and rng.random() < float(tear_cfg.get("probability", 1.0)):
        plan, tear_meta = sample_tear_plan(rgb_arr.shape[:2], tear_cfg, rng)
        rgb_arr = apply_tear_plan(rgb_arr, plan)
        if mask_arr is not None:
            mask_arr = apply_tear_plan(mask_arr, plan)
        _append_effect(applied, "tear", tear_meta)

    warp_cfg = aug_cfg.get("warp", {})
    if warp_cfg.get("enabled", False) and rng.random() < float(warp_cfg.get("probability", 1.0)):
        plan, warp_meta = sample_warp_plan(rgb_arr.shape[:2], warp_cfg, rng)
        if plan is not None:
            rgb_arr, mask_arr = apply_warp_plan(rgb_arr, mask_arr, plan, mask_mode=mask_mode)
            _append_effect(applied, "warp", warp_meta)

    lighting_cfg = aug_cfg.get("lighting", {})
    if lighting_cfg.get("enabled", False) and rng.random() < float(lighting_cfg.get("probability", 1.0)):
        rgb_arr, lighting_meta = apply_lighting(rgb_arr, lighting_cfg, rng)
        _append_effect(applied, "lighting", lighting_meta)

    blur_cfg = aug_cfg.get("blur", {})
    if blur_cfg.get("enabled", False) and rng.random() < float(blur_cfg.get("probability", 1.0)):
        rgb_arr, blur_meta = apply_blur(rgb_arr, blur_cfg, rng)
        if mask_arr is not None:
            mask_arr = apply_blur_meta(mask_arr, blur_meta)
        _append_effect(applied, "blur", blur_meta)

    photometric_fns = (
        ("chromatic_aberration", apply_chromatic_aberration),
        ("noise", apply_noise),
        ("jpeg", apply_jpeg),
        ("vignette", apply_vignette),
    )
    for name, fn in photometric_fns:
        effect_cfg = aug_cfg.get(name, {})
        rgb_arr, meta = _maybe_apply(rng, effect_cfg, fn, rgb_arr)
        if meta is not None:
            _append_effect(applied, name, meta)

    out_rgb = torch.from_numpy(_clip_rgb(rgb_arr)).to(device=rgb.device, dtype=rgb.dtype)
    out_mask: torch.Tensor | None = None
    if mask is not None and mask_arr is not None:
        out_mask = torch.from_numpy(mask_arr.astype(np.uint8)).to(device=mask.device, dtype=mask.dtype)
    return out_rgb, {"enabled": True, "applied": applied}, out_mask
