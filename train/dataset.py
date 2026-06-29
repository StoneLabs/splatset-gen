import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from config import cfg

ANNOTATION_FILES = ("annotations.jsonl", "annotations_processed.jsonl")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# On-the-fly train-time augmentation (not the generator's augment pipeline).
AUG = {
    "hflip_p": 0.50,
    "vflip_p": 0.50,
    "rotate_p": 0.50,
    "rotate_max_deg": 20.0,
    "jitter_p": 0.50,
    "jitter_brightness": 0.30,
    "jitter_contrast": 0.30,
    "jitter_saturation": 0.20,
    "crop_p": 0.30,
    "crop_min_scale": 0.80,
    "blur_p": 0.20,
}


def _resolve_dataset_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), _PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.is_dir():
            return candidate
    return (Path.cwd() / path).resolve()


def _annotation_path(dataset_dir: Path) -> Path | None:
    for name in ANNOTATION_FILES:
        path = dataset_dir / name
        if path.is_file():
            return path
    return None


def _mask_value_at(mask: Image.Image, x: float, y: float) -> int:
    px = int(round(x))
    py = int(round(y))
    px = max(0, min(mask.width - 1, px))
    py = max(0, min(mask.height - 1, py))
    return mask.getpixel((px, py))


def click_on_mask(mask: Image.Image, x: float, y: float, threshold: int = 127) -> bool:
    """Return True when the click lands on a foreground mask pixel."""
    return _mask_value_at(mask, x, y) > threshold


def _click_on_target_mask(mask_path: Path, point: list[int], threshold: int = 127) -> bool:
    """Return True when the annotated click is on a non-black mask pixel."""
    try:
        with Image.open(mask_path) as mask:
            return click_on_mask(
                mask.convert("L"),
                float(point[0]),
                float(point[1]),
                threshold=threshold,
            )
    except OSError:
        return False


def load_samples_from_dir(dataset_dir: Path, run_name: str | None = None) -> list[dict]:
    """Load samples from one generator output directory."""
    dataset_dir = dataset_dir.resolve()
    jsonl_path = _annotation_path(dataset_dir)
    if jsonl_path is None:
        return []

    run = run_name or dataset_dir.name
    samples: list[dict] = []
    total = 0
    skipped = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            mask_path = dataset_dir / rec["mask"]
            if not _click_on_target_mask(mask_path, rec["point"]):
                skipped += 1
                continue
            samples.append(
                {
                    "image": str(dataset_dir / rec["image"]),
                    "mask": str(mask_path),
                    "point": rec["point"],
                    "run": run,
                }
            )

    if total > 0 and not samples:
        raise ValueError(
            f"No valid samples in {dataset_dir}: all {skipped} annotation(s) have "
            "click on black mask or unreadable mask file"
        )
    if skipped:
        print(
            f"Warning: skipped {skipped}/{total} sample(s) in {dataset_dir} "
            "(click not on mask foreground)"
        )
    return samples


def load_all_samples(
    dataset_dirs: list[str | Path] | None = None,
    data_dir: str | Path | None = None,
    runs: list[str] | None = None,
) -> list[dict]:
    """
    Load annotated samples from explicit dataset directories or a legacy layout.

    Preferred: pass ``dataset_dirs`` — one entry per generator output, e.g.
    ``outputs/run_hq`` and ``outputs/adversial``. Shell expands globs before Python
    sees the arguments.

    Legacy: pass ``data_dir`` and optionally filter ``runs`` to ``run_*`` folders
    under that directory.
    """
    if dataset_dirs:
        samples: list[dict] = []
        for raw in dataset_dirs:
            path = _resolve_dataset_path(raw)
            if not path.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {path}")
            found = load_samples_from_dir(path)
            if not found:
                names = ", ".join(ANNOTATION_FILES)
                raise FileNotFoundError(f"No {names} found in {path}")
            samples.extend(found)
        return samples

    root = Path(data_dir or cfg.TRAINING_DATA_DIR).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {root}")

    selected = set(runs) if runs else None
    samples: list[dict] = []
    for entry in sorted(os.listdir(root)):
        run_path = root / entry
        if not (run_path.is_dir() and entry.startswith("run_")):
            continue
        if selected is not None and entry not in selected:
            continue
        samples.extend(load_samples_from_dir(run_path, run_name=entry))
    return samples


def stratified_split(samples, train_r, val_r, seed):
    """
    Split proportionally within each run so every split sees all distributions.
    Remainder after train+val goes to test.
    """
    rng = random.Random(seed)

    by_run = {}
    for s in samples:
        by_run.setdefault(s["run"], []).append(s)

    train, val, test = [], [], []
    for run_samples in by_run.values():
        rng.shuffle(run_samples)
        n = len(run_samples)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train += run_samples[:n_train]
        val += run_samples[n_train : n_train + n_val]
        test += run_samples[n_train + n_val :]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _image_center(width: int, height: int) -> tuple[float, float]:
    return (width - 1) / 2.0, (height - 1) / 2.0


def _rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
    """
    Rotate (x, y) counter-clockwise by angle_deg around (cx, cy).
    Matches PIL Image.rotate(..., expand=False) in image coordinates (y down).
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = x - cx
    dy = y - cy
    nx = cx + dx * cos_a + dy * sin_a
    ny = cy - dx * sin_a + dy * cos_a
    return nx, ny


def transform_hflip(
    img: Image.Image,
    mask: Image.Image,
    x: float,
    y: float,
) -> tuple[Image.Image, Image.Image, float, float]:
    width = img.width
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    x = (width - 1) - x
    return img, mask, x, y


def transform_vflip(
    img: Image.Image,
    mask: Image.Image,
    x: float,
    y: float,
) -> tuple[Image.Image, Image.Image, float, float]:
    height = img.height
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    y = (height - 1) - y
    return img, mask, x, y


def transform_rotate(
    img: Image.Image,
    mask: Image.Image,
    x: float,
    y: float,
    angle_deg: float,
) -> tuple[Image.Image, Image.Image, float, float]:
    width, height = img.size
    cx, cy = _image_center(width, height)
    img = img.rotate(angle_deg, resample=Image.BILINEAR, expand=False, center=(cx, cy))
    mask = mask.rotate(angle_deg, resample=Image.BILINEAR, expand=False, center=(cx, cy))
    x, y = _rotate_point(x, y, cx, cy, angle_deg)
    x = _clamp(x, 0.0, width - 1)
    y = _clamp(y, 0.0, height - 1)
    return img, mask, x, y


def transform_crop_resize(
    img: Image.Image,
    mask: Image.Image,
    x: float,
    y: float,
    scale: float,
    x0: int,
    y0: int,
) -> tuple[Image.Image, Image.Image, float, float] | None:
    """Crop, resize back to original size. Returns None if click would fall outside crop."""
    width, height = img.size
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    if crop_w > width or crop_h > height:
        return None
    if not (x0 <= x < x0 + crop_w and y0 <= y < y0 + crop_h):
        return None

    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    img = img.crop(box).resize((width, height), Image.BILINEAR)
    mask = mask.crop(box).resize((width, height), Image.BILINEAR)
    x = (x - x0) / crop_w * width
    y = (y - y0) / crop_h * height
    x = _clamp(x, 0.0, width - 1)
    y = _clamp(y, 0.0, height - 1)
    return img, mask, x, y


def _color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    brightness = AUG["jitter_brightness"]
    contrast = AUG["jitter_contrast"]
    saturation = AUG["jitter_saturation"]
    if brightness > 0:
        img = ImageEnhance.Brightness(img).enhance(
            1 + rng.uniform(-brightness, brightness)
        )
    if contrast > 0:
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-contrast, contrast))
    if saturation > 0:
        img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-saturation, saturation))
    return img


def apply_augment(
    img: Image.Image,
    mask: Image.Image,
    x: float,
    y: float,
    rng: random.Random | None = None,
    *,
    aug: dict | None = None,
    overrides: dict | None = None,
) -> tuple[Image.Image, Image.Image, float, float]:
    """
    Apply random train-time augmentations.

    Spatial transforms update image, mask, and click together. Color-only ops touch
    the RGB image only.
    """
    rng = rng or random.Random()
    settings = aug or AUG
    overrides = overrides or {}
    width, height = img.size

    def _enabled(name: str, prob: float) -> bool:
        if name not in overrides:
            return rng.random() < prob
        value = overrides[name]
        return bool(value) and value is not False

    if _enabled("hflip", settings["hflip_p"]):
        img, mask, x, y = transform_hflip(img, mask, x, y)

    if _enabled("vflip", settings["vflip_p"]):
        img, mask, x, y = transform_vflip(img, mask, x, y)

    if _enabled("rotate", settings["rotate_p"]):
        angle = overrides["rotate"] if isinstance(overrides.get("rotate"), (int, float)) else rng.uniform(
            -settings["rotate_max_deg"], settings["rotate_max_deg"]
        )
        img, mask, x, y = transform_rotate(img, mask, x, y, float(angle))

    if _enabled("crop", settings["crop_p"]):
        crop = overrides.get("crop")
        if isinstance(crop, dict):
            scale = float(crop["scale"])
            x0 = int(crop["x0"])
            y0 = int(crop["y0"])
        else:
            scale = rng.uniform(settings["crop_min_scale"], 1.0)
            crop_w = max(1, int(round(width * scale)))
            crop_h = max(1, int(round(height * scale)))
            max_x0 = min(int(x), width - crop_w)
            max_y0 = min(int(y), height - crop_h)
            x0 = rng.randint(0, max(0, max_x0))
            y0 = rng.randint(0, max(0, max_y0))
        cropped = transform_crop_resize(img, mask, x, y, scale, x0, y0)
        if cropped is not None:
            img, mask, x, y = cropped

    if _enabled("jitter", settings["jitter_p"]):
        img = _color_jitter(img, rng)

    if _enabled("blur", settings["blur_p"]):
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))

    return img, mask, x, y


class SplatDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        mask = Image.open(s["mask"]).convert("L")
        x, y = float(s["point"][0]), float(s["point"][1])
        W, H = img.size

        if self.augment:
            img, mask, x, y = apply_augment(img, mask, x, y)

        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask_t = mask_t.clamp(0.0, 1.0)
        pt = torch.tensor([x / (W - 1), y / (H - 1)], dtype=torch.float32)

        return img_t, pt, mask_t, s["run"]
