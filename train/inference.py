"""Load a trained checkpoint and predict segmentation masks from image + click.

Library API for viewer, scripts/predict.py, and tests. No CLI entry point here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from config import cfg
from model import PointConditionedUNet

OutputFormat = Literal["alpha", "binary"]
Visualization = Literal["raw", "compare"]
Background = Literal["transparent", "black"]

COMPARE_COLORS = {
    "tp": np.array([56, 203, 92], dtype=np.uint8),
    "fp": np.array([235, 64, 64], dtype=np.uint8),
    "fn": np.array([255, 255, 255], dtype=np.uint8),
}


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


def _validate_output_format(value: str) -> OutputFormat:
    text = value.lower()
    if text not in {"alpha", "binary"}:
        raise ValueError(f"Unknown output format {value!r}; use alpha or binary")
    return text  # type: ignore[return-value]


def _validate_visualization(value: str) -> Visualization:
    text = value.lower()
    if text not in {"raw", "compare"}:
        raise ValueError(f"Unknown visualization {value!r}; use raw or compare")
    return text  # type: ignore[return-value]


def _validate_background(value: str) -> Background:
    text = value.lower()
    if text not in {"transparent", "black"}:
        raise ValueError(f"Unknown background {value!r}; use transparent or black")
    return text  # type: ignore[return-value]


def classify_compare_pixels(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    *,
    binary: bool,
    cutoff: int,
) -> np.ndarray:
    """Return H×W uint8 kind map: 0=tn, 1=tp, 2=fp, 3=fn."""
    pred = pred_u8.astype(np.uint16)
    gt = gt_u8.astype(np.uint16)
    kinds = np.zeros(pred.shape, dtype=np.uint8)

    if binary:
        p = pred > cutoff
        g = gt > cutoff
        kinds[p & g] = 1
        kinds[~p & g] = 3
        kinds[p & ~g] = 2
        return kinds

    pred_a = pred / 255.0
    gt_a = gt / 255.0
    overlap = np.minimum(pred_a, gt_a)
    fn_amount = np.maximum(0.0, gt_a - pred_a)
    fp_amount = np.maximum(0.0, pred_a - gt_a)
    strength = np.maximum(np.maximum(overlap, fn_amount), fp_amount)
    signal = strength >= 0.01

    kinds[signal & (fp_amount >= overlap) & (fp_amount >= fn_amount)] = 2
    kinds[signal & (fn_amount >= overlap) & (fn_amount > fp_amount)] = 3
    kinds[signal & (kinds == 0)] = 1
    return kinds


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

    @torch.no_grad()
    def predict_mask(self, image_path: Path | str, point: list[int] | tuple[int, int]) -> np.ndarray:
        """Return binarized mask as uint8 0/255 (thresholded alpha)."""
        alpha = self.predict_alpha(image_path, point)
        return ModelRunner.binarize_alpha(alpha, self.mask_threshold)

    @staticmethod
    def binarize_alpha(alpha_u8: np.ndarray, threshold: float) -> np.ndarray:
        cutoff = int(round(threshold * 255))
        return np.where(alpha_u8 > cutoff, 255, 0).astype(np.uint8)

    @staticmethod
    def alpha_u8_to_rgba(alpha: np.ndarray) -> np.ndarray:
        """Pack soft alpha into RGBA PNG (white RGB, alpha = model confidence)."""
        encoded, _ = ModelRunner.encode_prediction_png(
            alpha,
            output_format="alpha",
            visualization="raw",
            background="transparent",
        )
        return encoded

    @staticmethod
    def encode_alpha_png(
        alpha: np.ndarray,
        *,
        background: str = "transparent",
    ) -> np.ndarray:
        """Backward-compatible alpha-only encoder."""
        encoded, _ = ModelRunner.encode_prediction_png(
            alpha,
            output_format="alpha",
            visualization="raw",
            background=background,
        )
        return encoded

    @staticmethod
    def alpha_png_mode(background: str) -> str:
        background = _validate_background(background)
        return "L" if background == "black" else "RGBA"

    @staticmethod
    def encode_compare_png(
        pred_u8: np.ndarray,
        gt_u8: np.ndarray,
        *,
        output_format: str = "alpha",
        background: str = "black",
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, str]:
        """Red/green/white error map (matches viewer compare mask rendering)."""
        output_format = _validate_output_format(output_format)
        background = _validate_background(background)
        if pred_u8.shape != gt_u8.shape:
            raise ValueError(f"pred/gt shape mismatch: {pred_u8.shape} vs {gt_u8.shape}")

        if output_format == "binary":
            cutoff = int(round(threshold * 255))
            pred_plane = ModelRunner.binarize_alpha(pred_u8, threshold)
            gt_plane = np.where(gt_u8 > cutoff, 255, 0).astype(np.uint8)
        else:
            cutoff = int(round(threshold * 255))
            pred_plane = pred_u8
            gt_plane = gt_u8

        kinds = classify_compare_pixels(
            pred_plane,
            gt_plane,
            binary=output_format == "binary",
            cutoff=cutoff,
        )
        height, width = pred_u8.shape
        strength = None
        if output_format == "alpha":
            pred_a = pred_plane.astype(np.float32) / 255.0
            gt_a = gt_plane.astype(np.float32) / 255.0
            overlap = np.minimum(pred_a, gt_a)
            fn_amount = np.maximum(0.0, gt_a - pred_a)
            fp_amount = np.maximum(0.0, pred_a - gt_a)
            strength = np.maximum(np.maximum(overlap, fn_amount), fp_amount)

        if background == "black":
            out = np.zeros((height, width, 3), dtype=np.uint8)
            for kind_id, color in ((1, COMPARE_COLORS["tp"]), (2, COMPARE_COLORS["fp"]), (3, COMPARE_COLORS["fn"])):
                mask = kinds == kind_id
                if not np.any(mask):
                    continue
                if strength is None:
                    out[mask] = color
                else:
                    out[mask] = np.clip(
                        np.round(strength[mask, None] * color),
                        0,
                        255,
                    ).astype(np.uint8)
            return out, "RGB"

        out = np.zeros((height, width, 4), dtype=np.uint8)
        for kind_id, color in ((1, COMPARE_COLORS["tp"]), (2, COMPARE_COLORS["fp"]), (3, COMPARE_COLORS["fn"])):
            mask = kinds == kind_id
            out[mask, 0] = color[0]
            out[mask, 1] = color[1]
            out[mask, 2] = color[2]
            if strength is None:
                out[mask, 3] = 255
            else:
                out[mask, 3] = np.clip(np.round(strength[mask] * 255.0), 0, 255).astype(np.uint8)
        return out, "RGBA"

    @staticmethod
    def encode_prediction_png(
        alpha_u8: np.ndarray,
        *,
        output_format: str = "alpha",
        visualization: str = "raw",
        background: str = "transparent",
        gt_u8: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, str]:
        """Encode a prediction for PNG export.

        Dimensions
        ----------
        output_format:
            ``alpha`` — soft sigmoid mask
            ``binary`` — thresholded white detect mask
        visualization:
            ``raw`` — mask output
            ``compare`` — TP green / FP red / FN white error map (needs ``gt_u8``)
        background:
            ``transparent`` — clear background (RGBA PNG)
            ``black`` — black background (L or RGB PNG)
        """
        if alpha_u8.ndim != 2:
            raise ValueError(f"expected HxW alpha, got shape {alpha_u8.shape}")

        output_format = _validate_output_format(output_format)
        visualization = _validate_visualization(visualization)
        background = _validate_background(background)

        if visualization == "compare":
            if gt_u8 is None:
                raise ValueError("compare visualization requires gt_u8")
            return ModelRunner.encode_compare_png(
                alpha_u8,
                gt_u8,
                output_format=output_format,
                background=background,
                threshold=threshold,
            )

        alpha = alpha_u8.astype(np.uint8, copy=False)
        if output_format == "binary":
            mask = ModelRunner.binarize_alpha(alpha, threshold)
            if background == "black":
                return mask, "L"
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = mask
            return rgba, "RGBA"

        if background == "black":
            return alpha, "L"
        rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = 255
        rgba[..., 2] = 255
        rgba[..., 3] = alpha
        return rgba, "RGBA"

    def predict_png(
        self,
        image_path: Path | str,
        point: list[int] | tuple[int, int],
        *,
        gt_path: Path | str | None = None,
        output_format: str = "alpha",
        visualization: str = "raw",
        background: str = "transparent",
    ) -> tuple[np.ndarray, str]:
        """Run model and return ``(array, pil_mode)`` ready for ``Image.fromarray``."""
        alpha = self.predict_alpha(image_path, point)
        gt_u8 = None
        if visualization.lower() == "compare":
            if gt_path is None:
                raise ValueError("compare visualization requires gt_path")
            gt_u8 = np.array(Image.open(gt_path).convert("L"))
        return self.encode_prediction_png(
            alpha,
            output_format=output_format,
            visualization=visualization,
            background=background,
            gt_u8=gt_u8,
            threshold=self.mask_threshold,
        )
