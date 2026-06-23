# CPU Gaussian-splat training data generator

Generates synthetic click-to-segment datasets from 3D Gaussian Splatting PLY objects.

## Setup

```bash
uv sync --extra dev
```

Requires Python 3.12+. PyTorch CPU wheel installed via `uv` (see `pyproject.toml`).

Place 3DGS `.ply` files in `assets/ply/`.

## Debug render (single PLY)

```bash
PYTHONPATH=src python scripts/render_debug.py \
  --ply assets/ply/Grape.ply \
  --max-gaussians 8000 \
  --width 512 --height 512 \
  --verbose
```

Omit `--max-gaussians` to load the full PLY.

## Generate dataset

Run with no arguments to use defaults (`assets/ply`, `configs/default.yaml`):

```bash
PYTHONPATH=src python scripts/generate_dataset.py
```

Shows a plan summary and waits for Enter. Use `-y` to skip confirmation.

```bash
PYTHONPATH=src python scripts/generate_dataset.py -h

PYTHONPATH=src python scripts/generate_dataset.py \
  -n 10 -j 1 -c configs/dev_fast.yaml -o outputs/test_run -y
```

Gaussian cap for testing: set `generation.max_gaussians_per_object` in config (see `configs/dev_fast.yaml`) — loads a **random subset** per PLY. Omit for full-quality runs.

Each sample writes:

- `images/000001.png` — composited RGB
- `masks/000001.png` — **visible-only** object mask (occluded regions black)
- `annotations.jsonl` — click `(x, y)`, object id, camera, scene metadata

Mask rule: `mask = (object_id_map == clicked_object_id)`. Occluder pixels get the front object's id in `object_id_map`, so rear-object masks are black where covered.

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Performance (CPU)

| Setting | Rough expectation |
|---|---|
| 512×512, ~300k Gaussians, 1 worker | ~2–5 min/sample |
| 4 workers | ~4× throughput (memory dependent) |
| 256×256 | ~4× faster |

Use `generation.max_gaussians_per_object` in config for faster dev iteration.

## v1 limits

- CPU-only rasterizer (adapted from EasyGaussianSplatting)
- SH degree 0 (DC color only)
- Solid background only (`background.mode: solid`); image backgrounds are v2
