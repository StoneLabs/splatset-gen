"""Load a trained checkpoint and predict segmentation masks from image + click."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import cfg
from model import PointConditionedUNet


def resolve_device(device: str | None = None) -> torch.device:
    setting = device or cfg.DEVICE
    if setting != "auto":
        return torch.device(setting)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path | str, device: torch.device | None = None) -> tuple[PointConditionedUNet, dict]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = device or resolve_device()
    model = PointConditionedUNet().to(dev)
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "checkpoint": str(path),
        "epoch": int(ckpt.get("epoch", 0)),
        "device": str(dev),
    }
    return model, meta


class ModelRunner:
    """Cached model for repeated inference (viewer, scripts)."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: str | None = None,
        mask_threshold: float | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model, self.meta = load_model(checkpoint_path, self.device)
        self.mask_threshold = float(mask_threshold if mask_threshold is not None else cfg.MASK_THRESHOLD)

    @torch.no_grad()
    def predict_alpha(self, image_path: Path | str, point: list[int] | tuple[int, int]) -> np.ndarray:
        """Return alpha mask as uint8 array with values 0–255 (sigmoid output)."""
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        img_t = (
            torch.from_numpy(np.array(img))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        ).to(self.device)
        pt = torch.tensor(
            [[point[0] / (width - 1), point[1] / (height - 1)]],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.model(img_t, pt)
        alpha = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        return (alpha * 255.0).astype(np.uint8)

    @staticmethod
    def alpha_u8_to_rgba(alpha: np.ndarray) -> np.ndarray:
        """Pack soft alpha into RGBA PNG (white RGB, alpha = model confidence)."""
        return ModelRunner.encode_alpha_png(alpha, background="transparent")

    @staticmethod
    def encode_alpha_png(
        alpha: np.ndarray,
        *,
        background: str = "transparent",
    ) -> np.ndarray:
        """Encode alpha mask for PNG export.

        ``transparent``: RGBA with white RGB and alpha = confidence (clear background).
        ``black``: grayscale L with white foreground on black (legacy mask PNG style).
        """
        if alpha.ndim != 2:
            raise ValueError(f"expected HxW alpha, got shape {alpha.shape}")
        alpha_u8 = alpha.astype(np.uint8, copy=False)
        if background == "black":
            return alpha_u8
        if background != "transparent":
            raise ValueError(f"Unknown alpha background {background!r}; use transparent or black")
        rgba = np.empty((*alpha_u8.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = 255
        rgba[..., 2] = 255
        rgba[..., 3] = alpha_u8
        return rgba

    @staticmethod
    def alpha_png_mode(background: str) -> str:
        return "L" if background == "black" else "RGBA"

    @torch.no_grad()
    def predict_mask(self, image_path: Path | str, point: list[int] | tuple[int, int]) -> np.ndarray:
        """Return binarized mask as uint8 0/255 (thresholded alpha)."""
        alpha = self.predict_alpha(image_path, point)
        return np.where(alpha > int(round(self.mask_threshold * 255)), 255, 0).astype(np.uint8)
