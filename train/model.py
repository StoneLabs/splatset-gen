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
    network explicit, exact localization.

    Computed at whatever (h, w) is asked for, so the same exact click signal can
    be re-injected at every encoder resolution — this is what keeps the spatial
    conditioning from being washed out by the deep conv stacks.
    """
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each

    px = pt[:, 0].view(-1, 1, 1, 1)   # per-sample click x, normalised [0, 1]
    py = pt[:, 1].view(-1, 1, 1, 1)   # per-sample click y, normalised [0, 1]

    dx = grid_x.view(1, 1, h, w) - px   # (B, 1, H, W)
    dy = grid_y.view(1, 1, h, w) - py
    return torch.cat([dx, dy], dim=1)   # (B, 2, H, W)


def _up_cat(x, skip):
    """Bilinear 2× upsample then concatenate with the encoder skip feature map."""
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([x, skip], dim=1)


def _channels_from_base(base_ch):
    """Graded 5-stage widths in the default proportions ([1, 2, 4, 6, 7]·base)."""
    return [base_ch, base_ch * 2, base_ch * 4, base_ch * 6, base_ch * 7]


# ── Main model ─────────────────────────────────────────────────────────────

class PointConditionedUNet(nn.Module):
    """
    U-Net conditioned on the click via explicit coordinate channels.

    Input:  RGB image       (B, 3, 512, 512)
            normalised point (B, 2)  ← exact float, never discretised
    Output: mask logits     (B, 1, 512, 512)  — sigmoid + threshold at inference

    Two deliberate choices target spatial understanding:

      • Depth.  The encoder has one stage per entry in `channels` (5 by default),
        so the bottleneck sits at 16×16 instead of 32×32. The extra downsample
        roughly doubles the receptive field, letting a single neuron near the
        bottleneck integrate context across the whole object rather than a local
        patch — the main lever for resolving where an object stops.

      • Persistent click signal.  The two click-offset channels are recomputed at
        each resolution and concatenated at *every* encoder stage (and the
        bottleneck), not just at the input. Deep layers therefore still know,
        exactly, where the click was — the localization cue no longer decays with
        depth. This is almost free in parameters (+2 input channels per block).

    Architecture is fully driven by `channels`; widen or deepen from config alone.
    `base_ch` is kept as a convenience override that derives the default graded
    proportions from a single number (used by the fast resume test).
    """
    def __init__(self, channels=None, base_ch=None):
        super().__init__()
        if channels is None:
            channels = _channels_from_base(base_ch) if base_ch else cfg.CHANNELS
        chs = list(channels)
        self.channels = chs
        L = len(chs)

        self.pool = nn.MaxPool2d(2)

        # encoder — every stage's first conv also ingests 2 click-offset channels
        self.enc = nn.ModuleList()
        for i, out_ch in enumerate(chs):
            prev = 3 if i == 0 else chs[i - 1]   # RGB at the top, features below
            self.enc.append(ConvBlock(prev + 2, out_ch))

        # bottleneck at the lowest resolution — coords re-injected here too
        self.bottleneck = ConvBlock(chs[-1] + 2, chs[-1])

        # decoder — dec[i] reconstructs encoder level i from up(prev) ⊕ skip[i].
        # Its input is up(prev)=chs[i] concatenated with skip e[i]=chs[i] → 2·chs[i];
        # it outputs chs[i-1] (or chs[0] at the top level).
        self.dec = nn.ModuleList()
        for i in range(L):
            out_ch = chs[i - 1] if i > 0 else chs[0]
            self.dec.append(ConvBlock(2 * chs[i], out_ch))

        # 1×1 projection to single-channel logits
        self.head = nn.Conv2d(chs[0], 1, 1)

    def _coords_like(self, pt, ref):
        return _coord_channels(pt, ref.shape[2], ref.shape[3], ref.device, ref.dtype)

    def forward(self, img, pt):
        # ── encode, re-injecting the exact click offsets at each resolution ──
        skips = []
        x = img
        for i, block in enumerate(self.enc):
            if i > 0:
                x = self.pool(x)
            x = torch.cat([x, self._coords_like(pt, x)], dim=1)
            x = block(x)
            skips.append(x)

        # ── bottleneck (16 × 16 at default depth) ────────────────────────────
        b = self.pool(skips[-1])
        b = torch.cat([b, self._coords_like(pt, b)], dim=1)
        d = self.bottleneck(b)

        # ── decode with skip connections, deepest level first ────────────────
        for i in reversed(range(len(self.dec))):
            d = self.dec[i](_up_cat(d, skips[i]))

        return self.head(d)   # (B, 1, 512, 512)
