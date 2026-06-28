import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# ── Building blocks ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard U-Net double conv: (conv → BN → ReLU) ×2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


def _coord_channels(pt, h, w, device, dtype):
    """
    Build two input channels encoding each pixel's signed offset from the click.
    dx, dy ∈ roughly [-1, 1]; both are exactly 0 at the clicked pixel, giving the
    network explicit, exact localization from the very first layer.
    """
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each

    px = pt[:, 0].view(-1, 1, 1, 1)   # per-sample click x, normalised [0, 1]
    py = pt[:, 1].view(-1, 1, 1, 1)   # per-sample click y, normalised [0, 1]

    dx = grid_x.view(1, 1, h, w) - px   # (B, 1, H, W)
    dy = grid_y.view(1, 1, h, w) - py
    return torch.cat([dx, dy], dim=1)   # (B, 2, H, W)


# ── Main model ─────────────────────────────────────────────────────────────

class PointConditionedUNet(nn.Module):
    """
    Plain U-Net conditioned on the click via two explicit coordinate channels.

    Input:  RGB image      (B, 3, 512, 512)
            normalised point (B, 2)  ← exact float, never discretised
    Output: mask logits    (B, 1, 512, 512)  — sigmoid + threshold at inference
    """
    def __init__(self, base_ch=None):
        super().__init__()
        b   = base_ch or cfg.BASE_CHANNELS
        chs = [b, b * 2, b * 4, b * 8]   # e.g. [32, 64, 128, 256]

        # encoder — first block takes RGB + 2 coordinate channels (5 in total)
        self.enc1 = ConvBlock(5,      chs[0])   # 512 × 512
        self.enc2 = ConvBlock(chs[0], chs[1])   # 256 × 256
        self.enc3 = ConvBlock(chs[1], chs[2])   # 128 × 128
        self.enc4 = ConvBlock(chs[2], chs[3])   # 64  × 64

        self.pool = nn.MaxPool2d(2)

        # bottleneck at the lowest resolution
        self.bottleneck = ConvBlock(chs[3], chs[3])   # 32 × 32

        # decoder — each block receives upsample(prev) ⊕ skip
        self.dec4 = ConvBlock(chs[3] + chs[3], chs[2])   # 64  × 64
        self.dec3 = ConvBlock(chs[2] + chs[2], chs[1])   # 128 × 128
        self.dec2 = ConvBlock(chs[1] + chs[1], chs[0])   # 256 × 256
        self.dec1 = ConvBlock(chs[0] + chs[0], chs[0])   # 512 × 512

        # 1×1 projection to single-channel logits
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, img, pt):
        # ── stack RGB with the exact click-offset channels ──────────────
        coords = _coord_channels(pt, img.shape[2], img.shape[3], img.device, img.dtype)
        x = torch.cat([img, coords], dim=1)   # (B, 5, 512, 512)

        # ── encode ──────────────────────────────────────────────────────
        e1 = self.enc1(x)                     # (B, 32,  512, 512)
        e2 = self.enc2(self.pool(e1))         # (B, 64,  256, 256)
        e3 = self.enc3(self.pool(e2))         # (B, 128, 128, 128)
        e4 = self.enc4(self.pool(e3))         # (B, 256, 64,  64)
        bn = self.bottleneck(self.pool(e4))   # (B, 256, 32,  32)

        # ── decode with skip connections ────────────────────────────────
        d4 = self.dec4(_up_cat(bn, e4))       # (B, 128, 64,  64)
        d3 = self.dec3(_up_cat(d4, e3))       # (B, 64,  128, 128)
        d2 = self.dec2(_up_cat(d3, e2))       # (B, 32,  256, 256)
        d1 = self.dec1(_up_cat(d2, e1))       # (B, 32,  512, 512)

        return self.head(d1)   # (B, 1, 512, 512)


def _up_cat(x, skip):
    """Bilinear 2× upsample then concatenate with the encoder skip feature map."""
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([x, skip], dim=1)
