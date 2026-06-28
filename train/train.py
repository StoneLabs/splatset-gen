import argparse
import os
import sys
import csv
import shutil
import signal
import time
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import cfg
from dataset import load_all_samples, stratified_split, SplatDataset
from model import PointConditionedUNet


# ── Ctrl+C ─────────────────────────────────────────────────────────────────

_stop = False

def _sigint(sig, frame):
    global _stop
    if _stop:
        print("\n\033[91m\033[1m force quit \033[0m\n")
        sys.exit(1)
    _stop = True
    print("\n\n\033[93m\033[1m ⚡ interrupt — finishing epoch then saving… \033[0m\n")

signal.signal(signal.SIGINT, _sigint)


# ── Device & AMP ───────────────────────────────────────────────────────────

def get_device():
    if cfg.DEVICE != "auto":
        return torch.device(cfg.DEVICE)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def print_device_report(device):
    """Startup debug: GPUs visible to PyTorch and which device training will use."""
    print(f"\n{_DIM}{'─' * 56}{_R}")
    print(f"{_BLD}  device report{_R}")
    print(f"  PyTorch {_DIM}{torch.__version__}{_R}  ·  CUDA built {_DIM}{torch.backends.cuda.is_built()}{_R}")
    print(f"  config device {_DIM}{cfg.DEVICE!r}{_R}")

    if torch.backends.mps.is_available():
        print(f"  MPS available {_GRN}yes{_R}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  CUDA available {_GRN}yes{_R}  ·  {n} GPU(s)")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            selected = device.type == "cuda" and (device.index in (None, i))
            tag = f"  {_GRN}{_BLD}← training{_R}" if selected else ""
            print(f"    [{i}] {props.name}  {mem_gb:.1f} GiB{tag}")
    else:
        print(f"  CUDA available {_RED}no{_R}")
        if not torch.backends.cuda.is_built():
            print(f"  {_YLW}hint: install CUDA-enabled PyTorch (see pyproject.toml){_R}")

    if device.type == "cpu":
        if cfg.DEVICE != "auto":
            reason = f"device={cfg.DEVICE!r} in config"
        elif torch.backends.mps.is_available():
            reason = "unexpected — MPS should have been selected"
        else:
            reason = "no CUDA GPU visible to PyTorch"
        print(f"  {_RED}{_BLD}training on CPU{_R}  {_DIM}({reason}){_R}")
    elif device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        print(f"  {_GRN}{_BLD}training on {device}{_R}  {_DIM}({name}){_R}")
    else:
        print(f"  {_GRN}{_BLD}training on {device}{_R}")
    print(f"{_DIM}{'─' * 56}{_R}\n")


def _amp_ctx(device):
    """
    AMP is CUDA-only. On MPS, float16 autocast triggers dtype-conversion sync
    points that serialise the pipeline and make training several times slower.
    """
    if cfg.USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ── ANSI palette ───────────────────────────────────────────────────────────

_R   = "\033[0m"
_BLD = "\033[1m"
_DIM = "\033[90m"   # dark gray
_GRN = "\033[92m"   # good / improving
_YLW = "\033[93m"   # warning / medium
_RED = "\033[91m"   # bad / critical
_CYN = "\033[96m"   # labels

BAR_W  = 30
LINE_W = 88


def _fmt_time(s):
    s = int(s)
    if s >= 3600: return f"{s//3600}h {(s%3600)//60}m"
    if s >= 60:   return f"{s//60}m {s%60}s"
    return f"{s}s"


def _c_metric(v, *, bold: bool = False):
    col = _GRN if v > 0.70 else (_YLW if v > 0.50 else _DIM)
    text = f"{col}{v:.3f}{_R}"
    return f"{_BLD}{text}{_R}" if bold else text


def _bar(ratio):
    n = int(BAR_W * ratio)
    return f"{_CYN}{'━'*n}{_R}{_DIM}{'━'*(BAR_W-n)}{_R}"


# ── Display ────────────────────────────────────────────────────────────────

def print_banner(device, model, n_train, n_val, n_test):
    params = sum(p.numel() for p in model.parameters())
    amp    = "amp" if (cfg.USE_AMP and device.type == "cuda") else "no amp"
    info   = f"  ◆ splat·ai   {device} · {params/1e6:.2f}M params · batch {cfg.BATCH_SIZE} · lr {cfg.LR} · {amp}"
    split  = f"  train {n_train} · val {n_val} · test {n_test} · seed {cfg.SEED}"
    w      = max(len(info), len(split)) + 2
    print(f"\n{_DIM}{'─'*w}{_R}")
    print(f"{_BLD}{info}{_R}")
    print(f"{_DIM}{split}{_R}")
    print(f"{_DIM}{'─'*w}{_R}\n")


def print_epoch_header(epoch, total, eta_str):
    print(f"\n{_CYN}{_BLD}[{epoch:03d}/{total}]{_R}  {_DIM}◷ {eta_str}{_R}")


def print_progress(batch, n_batches, running_loss, t0):
    ratio   = batch / max(n_batches, 1)
    elapsed = time.time() - t0
    eta     = (elapsed / batch) * (n_batches - batch) if batch > 0 else 0
    bar     = _bar(ratio)
    visible = f"  {'━'*BAR_W}  {ratio*100:5.1f}%  {batch}/{n_batches}  loss {running_loss:.4f}  eta {_fmt_time(eta)}"
    colored = f"  {bar}  {ratio*100:5.1f}%  {batch}/{n_batches}  loss {running_loss:.4f}  eta {_fmt_time(eta)}"
    print(f"\r{colored}{' ' * max(0, LINE_W - len(visible))}", end="", flush=True)


def print_epoch_done(train_loss, duration):
    bar     = f"{_GRN}{'━'*BAR_W}{_R}"
    visible = f"  {'━'*BAR_W}  loss {train_loss:.4f}   {_fmt_time(duration)}"
    colored = f"  {bar}  loss {train_loss:.4f}   {_DIM}{_fmt_time(duration)}{_R}"
    print(f"\r{colored}{' ' * max(0, LINE_W - len(visible))}")


def print_val(val_loss, agg, per_run):
    print(
        f"  {_DIM}val{_R}  {val_loss:.4f}"
        f"  ·  bin F1 {_c_metric(agg['bin_f1'])}"
        f"  ·  bin IoU {_c_metric(agg['bin_iou'])}"
        f"  ·  bin Dice {_c_metric(agg['bin_dice'])}"
    )
    print(
        f"  {_DIM}alpha{_R}  {_DIM}(loss target){_R}  "
        f"soft F1 {_c_metric(agg['soft_f1'], bold=True)}"
        f"  ·  soft IoU {_c_metric(agg['soft_iou'])}"
        f"  ·  MAE {_DIM}{agg['alpha_mae']:.3f}{_R}"
    )
    items = sorted(per_run.items())
    for i, (run, m) in enumerate(items):
        branch = f"{_DIM}└{_R}" if i == len(items) - 1 else f"{_DIM}├{_R}"
        print(
            f"  {branch}  {_DIM}{run:<20}{_R}"
            f"  bin F1 {_c_metric(m['bin_f1'])}"
            f"  soft F1 {_c_metric(m['soft_f1'], bold=True)}"
            f"  {_DIM}n={m['count']}{_R}"
        )


def print_status(is_best, saved_bkp, patience, max_patience, gap, lr):
    parts = []
    if is_best:
        parts.append(f"{_GRN}{_BLD}✦ best{_R}")
    if saved_bkp:
        parts.append(f"{_DIM}↺ backup{_R}")

    p_ratio = patience / max(max_patience, 1)
    p_str = (
        f"{_YLW}{_BLD}patience {patience}/{max_patience}{_R}" if p_ratio >= 0.5
        else f"{_DIM}patience {patience}/{max_patience}{_R}"
    )
    parts.append(p_str)
    parts.append(f"{_YLW}gap {gap:+.3f}{_R}" if gap > cfg.DIVERGENCE_GAP else f"{_DIM}gap {gap:+.3f}{_R}")
    parts.append(f"{_DIM}lr {lr:.2e}{_R}")
    print(f"  {'  ·  '.join(parts)}\n")


def print_best_checkpoint_saved(path, epoch, val_loss, prev_best_val_loss):
    name = os.path.basename(path)
    sidecar = checkpoint_config_path(path).name
    if prev_best_val_loss == float("inf"):
        val_detail = f"val {val_loss:.6f}  {_DIM}(first best){_R}"
        action = "saved"
    else:
        val_detail = f"val {prev_best_val_loss:.6f} → {val_loss:.6f}"
        action = "overwrote"
    print(
        f"  {_GRN}{_BLD}✦ {action} best checkpoint{_R}"
        f"  {_DIM}{name}{_R}"
        f"  ·  epoch {epoch}"
        f"  ·  {val_detail}"
        f"  ·  {_DIM}+ {sidecar}{_R}"
    )


def print_confusion(conf):
    """Render the aggregate pixel confusion matrix with row-normalised rates."""
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    total = tp + fp + fn + tn or 1.0
    def cell(v):                       # count + share of all pixels
        return f"{v:>13,.0f} {_DIM}({100*v/total:4.1f}%){_R}"
    print(f"\n  {_DIM}confusion (pixels){_R}")
    print(f"  {_DIM}{'':10}{'actual object':>26}{'actual background':>30}{_R}")
    print(f"  {_DIM}pred object{_R}     {_GRN}TP{_R} {cell(tp)}   {_RED}FP{_R} {cell(fp)}")
    print(f"  {_DIM}pred background{_R} {_RED}FN{_R} {cell(fn)}   {_GRN}TN{_R} {cell(tn)}")


# ── Loss & metrics ─────────────────────────────────────────────────────────

def bce_dice_loss(logits, target):
    """Soft alpha loss: BCE + soft Dice on continuous targets in [0, 1] (not bin F1)."""
    bce   = F.binary_cross_entropy_with_logits(logits, target)
    p     = torch.sigmoid(logits)
    pf    = p.view(p.size(0), -1)
    tf    = target.view(target.size(0), -1)
    inter = (pf * tf).sum(1)
    dice  = 1 - (2 * inter + 1) / (pf.sum(1) + tf.sum(1) + 1)
    return cfg.BCE_WEIGHT * bce + (1 - cfg.BCE_WEIGHT) * dice.mean()


# Binary metrics: pred and target thresholded at MASK_THRESHOLD.
BINARY_METRIC_KEYS = ["bin_iou", "bin_dice", "bin_f1", "bin_precision", "bin_recall"]
# Soft / alpha metrics: continuous sigmoid output vs soft target in [0, 1].
SOFT_METRIC_KEYS = ["soft_iou", "soft_f1", "soft_dice", "alpha_mae"]
METRIC_KEYS = BINARY_METRIC_KEYS + SOFT_METRIC_KEYS


def _batch_metrics_cpu(logits, masks):
    """
    Per-sample binary + soft alpha metrics and binary confusion counts.

    Binary: threshold prediction and target at ``MASK_THRESHOLD``.
    Soft: compare continuous alpha (sigmoid logits) to soft target values.
    ``soft_f1`` uses the soft Dice/F1 formula on alpha — equals ``soft_dice``.
    """
    alpha = torch.sigmoid(logits.float()).cpu()
    gt = masks.cpu().clamp(0.0, 1.0)
    d = (1, 2, 3)
    eps = 1e-6

    p_bin = (alpha > cfg.MASK_THRESHOLD).float()
    t_bin = (gt > cfg.MASK_THRESHOLD).float()
    tp = (p_bin * t_bin).sum(d)
    fp = (p_bin * (1 - t_bin)).sum(d)
    fn = ((1 - p_bin) * t_bin).sum(d)
    tn = ((1 - p_bin) * (1 - t_bin)).sum(d)
    smooth = 1.0
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    bin_metrics = {
        "bin_iou": (tp + smooth) / (tp + fp + fn + smooth),
        "bin_dice": (2 * tp + smooth) / (2 * tp + fp + fn + smooth),
        "bin_f1": 2 * prec * rec / (prec + rec),
        "bin_precision": prec,
        "bin_recall": rec,
    }

    inter = (alpha * gt).sum(d)
    alpha_sum = alpha.sum(d)
    gt_sum = gt.sum(d)
    union = alpha_sum + gt_sum - inter
    soft_dice = (2 * inter + eps) / (alpha_sum + gt_sum + eps)
    soft_metrics = {
        "soft_iou": (inter + eps) / (union + eps),
        "soft_dice": soft_dice,
        "soft_f1": soft_dice,
        "alpha_mae": (alpha - gt).abs().mean(d),
    }

    metrics = {**bin_metrics, **soft_metrics}
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return metrics, counts


# ── Train / eval ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    n          = len(loader)
    t0         = time.time()

    for i, (imgs, pts, masks, _) in enumerate(loader, 1):
        imgs, pts, masks = imgs.to(device), pts.to(device), masks.to(device)
        optimizer.zero_grad()

        with _amp_ctx(device):
            loss = bce_dice_loss(model(imgs, pts), masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        print_progress(i, n, total_loss / i, t0)

    return total_loss / n, time.time() - t0


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Returns mean loss, aggregate metrics, per-run breakdown, and a pixel confusion matrix."""
    model.eval()
    total_loss = 0.0
    agg        = {k: [] for k in METRIC_KEYS}   # batch-mean per metric
    per_run    = {}
    conf       = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}   # global pixel confusion

    for imgs, pts, masks, runs in loader:
        imgs, pts, masks = imgs.to(device), pts.to(device), masks.to(device)
        logits = model(imgs, pts)
        total_loss += bce_dice_loss(logits, masks).item()

        # pull to CPU once — avoids per-sample GPU slices accumulating in the allocator
        m, counts = _batch_metrics_cpu(logits, masks)
        del logits, imgs, pts, masks

        for k in conf:
            conf[k] += counts[k].sum().item()
        for k in METRIC_KEYS:
            agg[k].append(m[k].mean().item())

        for i, run in enumerate(runs):
            r = per_run.setdefault(run, {**{k: 0.0 for k in METRIC_KEYS}, "count": 0})
            for k in METRIC_KEYS:
                r[k] += m[k][i].item()
            r["count"] += 1

    for r in per_run.values():
        for k in METRIC_KEYS:
            r[k] /= r["count"]

    agg_mean = {k: float(np.mean(agg[k])) for k in METRIC_KEYS}
    return total_loss / len(loader), agg_mean, per_run, conf


# ── Checkpointing ──────────────────────────────────────────────────────────

def find_resume_checkpoint(checkpoint_dir):
    """Most recent training progress: interrupt → periodic backup → best."""
    for name, label in (
        (cfg.INTERRUPT_NAME, "interrupted"),
        (cfg.BACKUP_NAME, "backup"),
        (cfg.BEST_MODEL_NAME, "best"),
    ):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            return path, label
    return None, None


def checkpoint_config_path(checkpoint_path: str | os.PathLike) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}.config.yaml")


def save_checkpoint_config_snapshot(checkpoint_path: str | os.PathLike) -> None:
    config_src = Path(cfg.CONFIG_PATH)
    if not config_src.is_file():
        return
    dest = checkpoint_config_path(checkpoint_path)
    dest.write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")


def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    *,
    scheduler=None,
    scaler=None,
    training_state=None,
):
    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)
    save_checkpoint_config_snapshot(path)


def load_resume_checkpoint(path, model, optimizer, device, scaler=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt["epoch"]),
        ckpt.get("training_state", {}),
        ckpt.get("scheduler"),
    )


# ── Metrics logging ────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Appends one row per epoch to a CSV — ready to plot loss/IoU/Dice curves.
    Columns are fixed on the first write once the run names are known.
    """
    def __init__(self, path, *, append=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        resume = append and os.path.isfile(path) and os.path.getsize(path) > 0
        self._f      = open(path, "a" if resume else "w", newline="")
        self._writer = None
        self._runs   = None
        self._append = resume

    def log(self, epoch, lr, train_loss, val_loss, agg, per_run):
        if self._writer is None:
            self._runs = sorted(per_run)
            self._writer = csv.writer(self._f)
            if not self._append:
                header = ["epoch", "lr", "train_loss", "val_loss"]
                header += [f"val_{k}" for k in METRIC_KEYS]
                for r in self._runs:
                    header += [f"{r}_{k}" for k in METRIC_KEYS]
                self._writer.writerow(header)

        row = [epoch, f"{lr:.6g}", f"{train_loss:.6f}", f"{val_loss:.6f}"]
        row += [f"{agg[k]:.6f}" for k in METRIC_KEYS]
        for r in self._runs:
            row += [f"{per_run[r][k]:.6f}" for k in METRIC_KEYS]
        self._writer.writerow(row)
        self._f.flush()   # survive Ctrl+C and allow live plotting

    def close(self):
        self._f.close()


# ── Prediction export ──────────────────────────────────────────────────────

@torch.no_grad()
def save_predictions(model, loader, device, out_dir):
    model.eval()
    idx = 0
    for imgs, pts, _, runs in loader:
        imgs, pts = imgs.to(device), pts.to(device)
        preds = torch.sigmoid(model(imgs, pts)).float().cpu()
        for i in range(preds.size(0)):
            arr = (preds[i, 0].numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, f"{runs[i]}_{idx:05d}.png"))
            idx += 1


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train click-to-segment model on one or more splat generator outputs.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset directories, each with annotations.jsonl (shell expands globs, "
            "e.g. outputs/run_hq outputs/adversial or outputs/*). "
            "If omitted, scans config TRAINING_DATA_DIR for run_* folders."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete existing checkpoints, logs, and predictions before training (default: resume if checkpoint exists)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override training_config.yaml epochs (e.g. --epochs 1 for a quick smoke checkpoint)",
    )
    return parser.parse_args()


def _reset_training_artifacts() -> None:
    for path in (cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.PREDICTIONS_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)


def load_training_samples(dataset_dirs: list[str]) -> list[dict]:
    if dataset_dirs:
        return load_all_samples(dataset_dirs=dataset_dirs)
    return load_all_samples(data_dir=cfg.TRAINING_DATA_DIR, runs=cfg.TRAIN_RUNS or None)


def main():
    args = parse_args()
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    device = get_device()
    print_device_report(device)

    if args.restart:
        _reset_training_artifacts()
        print(f"{_YLW}restart — cleared checkpoints, logs, and predictions{_R}\n")

    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    if epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.epochs is not None:
        print(f"{_DIM}epochs override: {epochs} (config has {cfg.EPOCHS}){_R}\n")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    if cfg.SAVE_PREDICTIONS:
        if os.path.exists(cfg.PREDICTIONS_DIR):
            shutil.rmtree(cfg.PREDICTIONS_DIR)
        os.makedirs(cfg.PREDICTIONS_DIR)

    # ── Data ───────────────────────────────────────────────────────────
    samples = load_training_samples(args.datasets)
    if not samples:
        if args.datasets:
            print(f"{_RED}no samples found in {', '.join(args.datasets)}{_R}")
        else:
            print(f"{_RED}no samples found for runs {cfg.TRAIN_RUNS or 'all'}{_R}")
        return
    train_s, val_s, test_s = stratified_split(samples, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.SEED)
    runs_used = sorted({s["run"] for s in samples})

    pin    = (device.type == "cuda")
    ldr_kw = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                  pin_memory=pin, persistent_workers=(cfg.NUM_WORKERS > 0))

    # balance runs by inverse frequency so each contributes equally per epoch
    if cfg.BALANCE_RUNS and len(runs_used) > 1:
        run_counts = {r: sum(s["run"] == r for s in train_s) for r in runs_used}
        weights    = [1.0 / run_counts[s["run"]] for s in train_s]
        sampler    = WeightedRandomSampler(weights, num_samples=len(train_s), replacement=True)
        train_loader = DataLoader(SplatDataset(train_s, augment=True), sampler=sampler, **ldr_kw)
    else:
        train_loader = DataLoader(SplatDataset(train_s, augment=True), shuffle=True, **ldr_kw)

    val_loader  = DataLoader(SplatDataset(val_s),  shuffle=False, **ldr_kw)
    test_loader = DataLoader(SplatDataset(test_s), shuffle=False, **ldr_kw)

    # ── Model & optimiser ──────────────────────────────────────────────
    model     = PointConditionedUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg.USE_AMP) else None

    best_ckpt      = os.path.join(cfg.CHECKPOINT_DIR, cfg.BEST_MODEL_NAME)
    backup_ckpt    = os.path.join(cfg.CHECKPOINT_DIR, cfg.BACKUP_NAME)
    interrupt_ckpt = os.path.join(cfg.CHECKPOINT_DIR, cfg.INTERRUPT_NAME)

    resume_path, resume_kind = (None, None) if args.restart else find_resume_checkpoint(cfg.CHECKPOINT_DIR)
    resume_epoch = 0
    start_epoch  = 1
    best_val_loss     = float("inf")
    patience_counter  = 0
    divergence_streak = 0
    prev_train_loss   = float("inf")

    if resume_path:
        resume_epoch, ts, scheduler_state = load_resume_checkpoint(
            resume_path, model, optimizer, device, scaler=scaler,
        )
        start_epoch       = resume_epoch + 1
        best_val_loss     = ts.get("best_val_loss", float("inf"))
        patience_counter  = int(ts.get("patience_counter", 0))
        divergence_streak = int(ts.get("divergence_streak", 0))
        prev_train_loss   = float(ts.get("prev_train_loss", float("inf")))
    else:
        scheduler_state = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.LR_MIN, last_epoch=resume_epoch,
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    logger = MetricsLogger(
        os.path.join(cfg.LOG_DIR, "training_log.csv"),
        append=bool(resume_path),
    )

    print_banner(device, model, len(train_s), len(val_s), len(test_s))
    print(f"  {_DIM}runs: {', '.join(runs_used)}{_R}\n")
    if resume_path:
        print(
            f"  {_GRN}{_BLD}resume{_R}  {_DIM}{resume_kind}{_R} checkpoint"
            f"  ·  completed epoch {resume_epoch}"
            f"  ·  continuing {start_epoch}–{epochs}\n"
        )

    def _training_state():
        return {
            "best_val_loss":     best_val_loss,
            "patience_counter":  patience_counter,
            "divergence_streak": divergence_streak,
            "prev_train_loss":   prev_train_loss,
        }

    def _checkpoint_kwargs():
        return {
            "scheduler": scheduler,
            "scaler": scaler,
            "training_state": _training_state(),
        }

    # ── Loop ───────────────────────────────────────────────────────────
    epoch_times: deque = deque(maxlen=5)
    last_epoch = resume_epoch
    if start_epoch > epochs:
        print(f"  {_DIM}already at epoch {resume_epoch}/{epochs} — skipping training loop{_R}\n")
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        eta_str    = (
            f"~{_fmt_time(np.mean(epoch_times) * (epochs - epoch + 1))} left"
            if epoch_times else "estimating…"
        )
        print_epoch_header(epoch, epochs, eta_str)

        train_loss, duration = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_agg, val_per_run, _ = eval_epoch(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_times.append(duration)

        logger.log(epoch, current_lr, train_loss, val_loss, val_agg, val_per_run)

        is_best   = val_loss < best_val_loss
        saved_bkp = (epoch % cfg.BACKUP_EVERY_EPOCHS == 0)

        print_epoch_done(train_loss, duration)
        print_val(val_loss, val_agg, val_per_run)

        if is_best:
            prev_best_val_loss = best_val_loss
            best_val_loss      = val_loss
            patience_counter   = 0
            save_checkpoint(model, optimizer, epoch, best_ckpt, **_checkpoint_kwargs())
            print_best_checkpoint_saved(best_ckpt, epoch, val_loss, prev_best_val_loss)
        else:
            patience_counter += 1

        if saved_bkp:
            save_checkpoint(model, optimizer, epoch, backup_ckpt, **_checkpoint_kwargs())

        gap = val_loss - train_loss
        if gap > cfg.DIVERGENCE_GAP and train_loss < prev_train_loss:
            divergence_streak += 1
            if divergence_streak >= cfg.DIVERGENCE_STREAK:
                print(f"  {_RED}{_BLD}⚠ overfit{_R}  val/train gap={gap:.4f} for {divergence_streak} epochs")
        else:
            divergence_streak = 0
        prev_train_loss = train_loss

        print_status(is_best, saved_bkp, patience_counter, cfg.PATIENCE, gap, current_lr)

        if patience_counter >= cfg.PATIENCE:
            print(f"  {_YLW}soft stop — no improvement for {cfg.PATIENCE} epochs{_R}\n")
            break

        if _stop:
            save_checkpoint(model, optimizer, epoch, interrupt_ckpt, **_checkpoint_kwargs())
            print(f"  {_YLW}interrupted at epoch {epoch} — saved to {interrupt_ckpt}{_R}\n")
            break

    logger.close()

    # ── Test evaluation ────────────────────────────────────────────────
    if not os.path.exists(best_ckpt):
        print(f"{_YLW}no checkpoint found — skipping test evaluation{_R}\n")
        return

    print(f"{_DIM}{'─'*56}{_R}")
    print(f"{_CYN}{_BLD}  test evaluation{_R}  {_DIM}(best: epoch {torch.load(best_ckpt, weights_only=True)['epoch']} of {last_epoch}){_R}\n")

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    test_loss, test_agg, test_per_run, test_conf = eval_epoch(model, test_loader, device)
    print(
        f"  {_DIM}result{_R}  {test_loss:.4f}"
        f"  ·  bin F1 {_c_metric(test_agg['bin_f1'])}"
        f"  ·  bin IoU {_c_metric(test_agg['bin_iou'])}"
        f"  ·  bin Dice {_c_metric(test_agg['bin_dice'])}"
    )
    print(
        f"  {_DIM}alpha{_R}  {_DIM}(loss target){_R}  "
        f"soft F1 {_c_metric(test_agg['soft_f1'], bold=True)}"
        f"  ·  soft IoU {_c_metric(test_agg['soft_iou'])}"
        f"  ·  MAE {_DIM}{test_agg['alpha_mae']:.3f}{_R}"
    )
    items = sorted(test_per_run.items())
    for i, (run, m) in enumerate(items):
        branch = f"{_DIM}└{_R}" if i == len(items) - 1 else f"{_DIM}├{_R}"
        print(
            f"  {branch}  {_DIM}{run:<20}{_R}"
            f"  bin F1 {_c_metric(m['bin_f1'])}"
            f"  soft F1 {_c_metric(m['soft_f1'], bold=True)}"
            f"  {_DIM}n={m['count']}{_R}"
        )
    print_confusion(test_conf)

    if cfg.SAVE_PREDICTIONS:
        print(f"\n  {_DIM}writing masks to '{cfg.PREDICTIONS_DIR}/' …{_R}")
        save_predictions(model, test_loader, device, cfg.PREDICTIONS_DIR)
        print(f"  {_GRN}done{_R}")

    print(f"\n{_DIM}{'─'*56}{_R}\n")


if __name__ == "__main__":
    main()
