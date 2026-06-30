# splat-proj

CPU pipeline → synthetic click-to-segment data from 3DGS PLYs. Train U-Net. Run inference via CLI or viewer.

Each sample: RGB image, occlusion-aware object mask, click `(x, y)` in `annotations.jsonl`.

## Setup

```bash
uv sync --extra dev
```

Put 3DGS `.ply` files in `assets/ply/`. PyTorch installs CUDA wheel from `pyproject.toml` (falls back to CPU/MPS if no GPU).

## Configuration

| Area | Location | Notes |
|------|----------|-------|
| Dataset generation | `configs/` | Pass with `-c`; default `configs/default.yaml`. Also `dev_fast.yaml`, `high_quality.yaml`, `production.yaml`, … |
| Training & inference | `train/training_config.yaml` | Paths, hyperparams, default checkpoint. Inference section used by viewer and `scripts/predict.py`. |

Each generated run also snapshots its datagen config to `outputs/<run>/config.yaml`.

## Dataset generation

Renders multi-object scenes, samples click on foreground, writes masks from `object_id_map` (occluded pixels stay black).

```bash
uv run scripts/generate_dataset.py
uv run scripts/generate_dataset.py -o outputs/run2 -n 100 -c configs/dev_fast.yaml -y
uv run scripts/generate_dataset.py -o outputs/run2 --continue -n 50 -y
```

| Flag | Purpose |
|------|---------|
| `-o` | Output dir |
| `-n` | Sample count |
| `-j` | Workers (default: CPU−1) |
| `-c` | Datagen config (see `configs/`) |
| `--continue` | Append to existing run |
| `-y` | Skip confirmation |

Output per run:

```
outputs/run2/
├── config.yaml
├── images/000001.png
├── masks/000001.png
└── annotations.jsonl
```

## Viewer

Browse datasets. Optional AI panel when checkpoint loaded.

```bash
uv run viewer/app.py outputs
uv run viewer/app.py outputs --model train/checkpoints/best_by_val_loss.pth
```

Open **http://127.0.0.1:8765**. Dataset mode: RGB, mask, overlay, JSON, config. **Run AI** predicts mask for current sample. Interactive tab: upload image, click to predict.

## Training

Point-conditioned U-Net on generated datasets. See `train/training_config.yaml`.

```bash
uv run train/train.py outputs/run_hq outputs/run2
uv run train/train.py                    # all run_* under outputs/
uv run train/train.py --restart          # wipe checkpoints/logs
uv run train/train.py --epochs 1         # smoke test
```

Checkpoints → `train/checkpoints/` (`best_by_val_loss.pth`). Logs → `train/logs/training_log.csv`.

## Inference

### Script

```bash
uv run scripts/predict.py image.png 120 340 -o mask.png -c train/checkpoints/best_by_val_loss.pth
uv run scripts/predict.py image.png 64 64 -o compare.png -c ckpt.pth --visualization compare --gt gt.png
```

Checkpoint defaults to `inference.checkpoint` in `train/training_config.yaml`. Formats: `--format alpha|binary`, viz: `--visualization raw|compare`.

### Web (viewer)

```bash
uv run viewer/app.py outputs --model train/checkpoints/best_by_val_loss.pth
```

- **Dataset tab:** Run AI on current sample (uses stored click).
- **Interactive tab:** Upload image, click point → live prediction via `/api/predict/interactive`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_dataset.py` | Batch synthetic dataset |
| `scripts/render_debug.py` | Single PLY → debug PNG |
| `scripts/predict.py` | CLI inference (image + click → mask) |
| `scripts/plot_training_log.py` | Plot metrics from `train/logs/training_log.csv` |

## Tests

```bash
uv run pytest tests/ -v
```
