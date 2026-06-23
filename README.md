# splat-dataset

CPU-only pipeline that generates synthetic **click-to-segment** training data from 3D Gaussian Splatting (3DGS) PLY objects.

Each sample produces:

| Output | Description |
|--------|-------------|
| **RGB image** | Scene render (multi-object splats composited over background) |
| **Binary mask** | **Visible** portion of the clicked object only — occluded pixels are black |
| **Click `(x, y)`** | Stored in `annotations.jsonl` |

Object-level semantics: one PLY = one object instance. Masks are **modal** (not amodal): if object A is behind object B, A's mask is black where B covers it.

---

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/)
- 3DGS `.ply` files in `assets/ply/` (standard Inria format: `x,y,z`, `scale_*`, `rot_*`, `f_dc_*`, `opacity`, …)

---

## Setup

```bash
uv sync --extra dev
```

PyTorch is installed from the CPU wheel index (see `pyproject.toml`). No CUDA required.

---

## Generate a dataset

Defaults: `assets/ply/`, `configs/default.yaml`, `outputs/run_001/`, 100 samples.

```bash
uv run scripts/generate_dataset.py
```

Shows a pre-flight plan and waits for **Enter**. Skip confirmation with `-y`:

```bash
uv run scripts/generate_dataset.py --output outputs/run2/ -n 7 -c configs/dev_fast.yaml --seed 42 -v -y
```

### Common flags

| Flag | Short | Description |
|------|-------|-------------|
| `--output` | `-o` | Output directory |
| `--num-samples` | `-n` | Number of samples |
| `--workers` | `-j` | Parallel workers (default: CPU count − 1) |
| `--config` | `-c` | YAML config path |
| `--seed` | | Master RNG seed |
| `--verbose` | `-v` | Rasterizer progress in the Recent log panel |
| `--yes` | `-y` | Skip confirmation prompt |
| `--ply-dir` | | PLY input directory |
| `-h` | | Help |

Full help:

```bash
uv run scripts/generate_dataset.py -h
```

### Output layout

```
outputs/run2/
├── config.yaml           # snapshot of config used
├── images/000001.png     # composited RGB
├── masks/000001.png      # object mask (L mode, 0 or 255)
└── annotations.jsonl     # one JSON record per line
```

Example JSONL record:

```json
{
  "id": "000001",
  "image": "images/000001.png",
  "mask": "masks/000001.png",
  "point": [128, 256],
  "object_id": 1,
  "num_objects": 3,
  "background": { "mode": "solid", "color": [0.1, 0.1, 0.1] },
  "camera": { "width": 512, "height": 512, "fov_deg": 60.0, "viewmat": [...], "K": [...] },
  "objects": [{ "object_id": 0, "ply": "Grape.ply", "transform": { ... } }]
}
```

---

## Browse a dataset

Local web viewer for inspecting generated runs. Opens a 2×2 layout:

| Panel | Content |
|-------|---------|
| Top left | RGB render with red crosshair at click `(x, y)` |
| Top right | Object mask |
| Bottom left | Annotation JSON for the current sample |
| Bottom right | `config.yaml` snapshot from the run |

```bash
uv run viewer/app.py --dataset outputs/run2
```

Then open **http://127.0.0.1:8765** in a browser.

### Viewer flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `outputs/run2` | Dataset directory (must contain `annotations.jsonl`) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Bind port |
| `--debug` | off | Flask debug mode |

### UI controls

- **Sample navigation:** prev/next buttons, index input, ID jump, keyboard `←`/`→` (or `j`/`k`), `Home`/`End`
- **Fit images to panel:** checkbox (default on) — scale images to the panel; off shows native pixel size with scroll
- **Resizable panels:** drag the horizontal or vertical splitters between panels; layout ratios persist in browser `localStorage`

Large datasets (100k+ samples) use a byte-offset index on `annotations.jsonl` with an on-disk cache (`.viewer_index_*.pkl` in the dataset dir) so startup and random access stay fast without loading all annotations into memory.

---

## Configuration

Configs live in `configs/`. Key sections:

```yaml
render:
  width: 512
  height: 512
  alpha_threshold: 0.5
  sh_degree: 0              # v1: DC color only

background:
  mode: "solid"
  solid_color: [0.1, 0.1, 0.1]

scene:
  num_objects_min: 2
  num_objects_max: 5
  position_range: [-2.0, 2.0]
  rotation_deg_max: 180
  scale_jitter: [0.8, 1.2]

camera:
  fov_deg_range: [45, 75]
  distance_range: [3.0, 8.0]
  max_retries: 20

generation:
  max_camera_retries: 20
  max_gaussians_per_object: 25000   # optional; random subset per PLY for testing
```

- **`configs/default.yaml`** — full-quality settings (no Gaussian cap by default).
- **`configs/dev_fast.yaml`** — lower resolution + Gaussian cap for quick iteration.

When `generation.max_gaussians_per_object` is set, each PLY load takes a **random subset** of that many Gaussians (reproducible per seed). Omit the key for full PLYs.

---

## Debug render (single PLY)

Verify a PLY renders correctly before generating a full dataset:

```bash
uv run scripts/render_debug.py --ply assets/ply/Grape.ply --verbose
```

Omit `--max-gaussians` to load the entire PLY. Writes composited RGB and a foreground-only `_fg` PNG under `outputs/debug/`.

---

## How masks work

Rendering produces an `object_id_map`: per pixel, the object ID of the **dominant visible** splat (front-to-back compositing). The click is sampled on foreground pixels; the mask is:

```
mask = (object_id_map == clicked_object_id)
```

Occluded regions of the clicked object receive the occluder's ID, so they stay black in the mask. No second render pass needed.

---

## Tests

```bash
uv run pytest tests/ -v
```

Includes PLY loading, camera math, single-object masks, and a synthetic two-object occlusion test.

---

## Performance (CPU)

Rough expectations on a typical desktop:

| Setting | Time / sample |
|---------|----------------|
| 512×512, ~300k Gaussians, 1 worker | ~1–5 min |
| 256×256 or capped Gaussians (`dev_fast`) | much faster |
| 4 workers | ~4× throughput (memory dependent) |

Correctness first; use `configs/dev_fast.yaml` or `max_gaussians_per_object` while iterating.

---

## Project layout

```
assets/ply/              # input 3DGS PLY objects
configs/                 # YAML scene/render settings
scripts/
  generate_dataset.py    # main CLI
  render_debug.py        # single-PLY debug render
viewer/
  app.py                 # dataset browser (Flask)
src/
  ply_loader.py          # PLY → SceneGaussians
  scene.py               # random multi-object placement
  camera.py              # pinhole camera sampling
  render/                # CPU splat rasterizer (EasyGS-derived)
  picker.py              # click sampling + occlusion-aware mask
  sample.py              # one-sample pipeline
  parallel.py            # multiprocessing + live progress UI
outputs/                 # generated datasets (gitignored)
```

---

## v1 scope

- CPU-only PyTorch rasterizer
- SH degree 0 (DC color)
- Solid background only (`background.mode: solid`)
- Visible (modal) object masks with occlusion

Not included: GPU rendering, background images, amodal masks, model training code.
