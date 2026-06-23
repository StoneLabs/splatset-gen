# Implementation Plan: CPU Gaussian-Splat Training Data Generator

> Hand this file to Composer 2.5 (or any agent) to implement the project from scratch.
> The workspace is empty today — this is the sole specification.

---

## 1. Goal

Build a **CPU-only** batch pipeline that generates synthetic training data for a **click-to-segment** model (SAM-style):

| Output | Description |
|---|---|
| **(a) RGB image** | Full scene render (all splat objects, composited over background) |
| **(b) Binary mask** | **Visible** portion of the clicked **object only** — occluded parts are **black** |
| **(c) Click coordinate** | `(x, y)` pixel stored as metadata |

**Object-level** semantics: each input PLY = one object instance. The click selects an object, not an individual Gaussian.

**Occlusion rule:** If object A is partially hidden behind object B, the mask for A must be **black** where B covers A. The mask reflects **visible** object pixels only (modal mask), not the full unoccluded silhouette.

**Background rule (v1):** Solid color only. Architecture must support random background **images** in v2 without changing the rasterizer, mask logic, or click sampling.

---

## 2. Stack (CPU-only)

| Layer | Choice | Rationale |
|---|---|---|
| Language | **Python 3.11+** | ML ecosystem, easy dataset tooling |
| Arrays / scene math | **NumPy** | Transforms, camera sampling |
| Rendering | **PyTorch CPU** + **vendored pure-PyTorch splat rasterizer** | gsplat/CUDA unavailable; need full control over per-pixel object ID |
| Rasterizer source | **Adapt from [gsplat-pytorch](https://github.com/Mxbonn/gsplat-pytorch)** or **[EasyGaussianSplatting `forward_cpu.py`](https://github.com/scomup/EasyGaussianSplatting)** | Both are pure PyTorch, no CUDA; vendoring avoids fragile tiny deps |
| PLY I/O | **`plyfile`** | Standard 3DGS PLY format |
| Image I/O | **Pillow** | PNG read/write |
| CLI | **Typer** | Simple CLI for generation |
| Config | **YAML** (`PyYAML`) | Scene/render parameters |
| Parallelism | **`multiprocessing`** | CPU-bound; parallelize across samples |
| Package manager | **`uv`** or **`pip`** + **`pyproject.toml`** | Standard Python packaging |
| Tests | **`pytest`** | Unit tests for loader, camera, mask logic |

### Explicitly do NOT use

- **gsplat** (CUDA-only for production paths)
- **Original Inria diff-gaussian-rasterization** (CUDA)
- **OpenSplat** as primary (no Python bindings; would require C++ wrapper)
- **Unity / Blender / WebGPU** (overkill, poor fit for batch pipeline)
- **SAM** for label generation (ground truth comes from 3D)

### PyTorch install (CPU only)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Core Algorithm

### 3.1 Scene representation

Each scene is a concatenation of loaded PLY objects:

```python
@dataclass
class SceneGaussians:
    means:      FloatTensor  # [N, 3]
    quats:      FloatTensor  # [N, 4]  wxyz
    scales:     FloatTensor  # [N, 3]  (exp-activated in PLY)
    opacities:  FloatTensor  # [N]     (sigmoid-activated in PLY)
    sh_dc:      FloatTensor  # [N, 3]  degree-0 SH (RGB)
    sh_rest:    FloatTensor  # [N, K, 3] higher-order SH (optional, v1 can use DC only)
    object_ids: IntTensor     # [N]     which PLY each Gaussian belongs to
```

On load, assign `object_ids` as contiguous integers `0..M-1` per PLY file.

Apply per-object random SE(3) transform (translation, rotation, optional scale jitter) **before** concatenation.

### 3.2 Single-pass splat render (foreground layer)

One rasterization pass outputs **foreground buffers** (before background composite):

1. **fg_rgb** `[H, W, 3]` — alpha-composited splat color over black
2. **alpha** `[H, W]` — accumulated splat opacity (0 = no splat, 1 = opaque splat)
3. **object_id_map** `[H, W]` — integer object ID of the **dominant visible** surface per pixel

**Dominant object ID rule** (during front-to-back alpha compositing per pixel):

```python
T = 1.0
color = background  # black during splat pass
dominant_oid = BACKGROUND_ID  # -1
best_weight = 0.0

for gaussian in gaussians_affecting_pixel_sorted_by_depth:
    alpha = gaussian_opacity * gaussian_kernel_response
    weight = alpha * T
    if weight > best_weight:
        best_weight = weight
        dominant_oid = gaussian.object_id
    color += weight * gaussian_color
    T *= (1.0 - alpha)
    if T < 1e-4:
        break

object_id_map[pixel] = dominant_oid
fg_rgb[pixel] = color
alpha[pixel] = 1.0 - T
```

This ensures occluded parts of an object get a **different** `dominant_oid` (the occluder's ID), so masks are automatically occlusion-aware.

`object_id_map` and `alpha` are computed **before** any background composite. Background pixels never receive a splat `object_id`.

### 3.3 Background compositing (v1: solid color)

Composite happens **outside** the rasterizer, in `background.py`:

```python
# v1
rgb = fg_rgb * alpha + solid_color * (1 - alpha)

# v2 (future — same function, different background argument)
rgb = fg_rgb * alpha + background_image * (1 - alpha)
```

Background images sit **behind all splats** (infinite depth). No depth ordering between background and splats.

### 3.4 Click selection

1. Render scene → `fg_rgb`, `alpha`, `object_id_map`
2. Composite → `rgb` via `background.py`
3. Sample candidate click `(x, y)` from **foreground splat pixels** only: `alpha[x, y] > alpha_threshold` (default `0.5`) and `object_id_map[x, y] >= 0`
4. `clicked_object_id = object_id_map[x, y]`
5. Reject/resample if no valid foreground pixels exist

Do **not** uniformly sample all pixels — most would land on background.
Do **not** sample clicks from final composited RGB — always use `alpha` + `object_id_map`.

### 3.5 Occlusion-aware object mask

```python
mask = (object_id_map == clicked_object_id).astype(uint8) * 255
```

Properties:

- White = visible pixels belonging to clicked object
- Black = background, other objects, **and occluded parts** of clicked object
- No second render pass needed for masks
- Background composite does not affect the mask

Optional sanity check: `mask[y, x] == 255` must always hold at the click pixel.

### 3.6 RGB image (output a)

Use the composited `rgb` buffer. Save as PNG.

---

## 4. Background Images (v2 — design for now)

v1 implements solid color only. The following constraints ensure v2 is a small addition, not a rewrite.

### v1 requirements (forward-compatible)

- `RenderOutput` separates `fg_rgb`, `alpha`, and composited `rgb`
- Compositing lives in `background.py`, not inside the tile rasterizer loop
- Config has a `background` section with `mode: "solid"`; image mode documented but unimplemented
- Masks and clicks use `object_id_map` / `alpha` from the splat pass, never post-composite RGB

### v2 additions (not in v1 scope)

```
assets/
├── ply/           # splat objects
└── backgrounds/   # JPG/PNG background library
```

```yaml
background:
  mode: "image"
  image_dir: "assets/backgrounds/"
  resize_mode: "crop"       # crop | letterbox | stretch
  # optional: color jitter, blur
```

| Concern | v2 behavior |
|---|---|
| RGB image | Random background + alpha-composited splats |
| Mask | Unchanged — `(object_id_map == clicked_object_id)` |
| Click sampling | Unchanged — sample only where `alpha > threshold` |
| Occlusion | Unchanged — occluder object IDs overwrite rear object IDs |
| Splat edges | Semi-transparent pixels blend RGB with background; `object_id_map` uses dominant splat weight |

Optional v2 debug export: `foreground/000001.png` (splat layer only, no background).

### What v1 must NOT do (would block v2)

- Do **not** bake background color into the rasterizer
- Do **not** compute masks after background composite
- Do **not** sample clicks from final RGB
- Do **not** merge `fg_rgb` and background inside the tile rasterizer loop

---

## 5. Project Structure

```
splat-proj/
├── PLAN.md                     # this file
├── pyproject.toml
├── README.md
├── configs/
│   └── default.yaml
├── assets/
│   ├── ply/                    # user-provided splat PLY files (gitkeep)
│   └── backgrounds/            # v2: background images (gitkeep, empty in v1)
├── src/
│   └── splat_dataset/
│       ├── __init__.py
│       ├── ply_loader.py       # load 3DGS PLY → SceneGaussians
│       ├── scene.py            # random placement, camera sampling
│       ├── camera.py           # intrinsics/extrinsics, look-at, random orbit
│       ├── background.py       # solid compositing (v1); image compositing (v2)
│       ├── render/
│       │   ├── __init__.py
│       │   ├── projection.py   # 3D → 2D projection, conic computation
│       │   ├── rasterizer.py   # tile-based CPU rasterizer (vendored/adapted)
│       │   └── sh.py           # SH evaluation (v1: DC term only)
│       ├── picker.py           # foreground pixel sampling + click validation
│       ├── sample.py           # orchestrates one sample: scene → render → pick → export
│       ├── export.py           # write PNG + JSON metadata
│       └── parallel.py         # multiprocessing worker pool
├── scripts/
│   └── generate_dataset.py     # Typer CLI entry point
├── tests/
│   ├── test_ply_loader.py
│   ├── test_camera.py
│   ├── test_mask_occlusion.py
│   └── test_single_object.py
└── outputs/                    # generated datasets (gitignored)
```

---

## 6. Module Specifications

### 6.1 `ply_loader.py`

- Load standard **3D Gaussian Splatting PLY** format
- Expected vertex fields: `x,y,z`, `nx,ny,nz` (optional), `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`
- Apply activations on load:
  - `opacity = sigmoid(raw_opacity)`
  - `scale = exp(raw_scale)`
  - normalize quaternion
- Return a single-object `SceneGaussians` with all `object_ids = 0`
- Raise clear error if PLY format doesn't match

### 6.2 `scene.py`

**`build_random_scene(ply_paths, config) -> SceneGaussians, metadata`**

Config parameters:

- `num_objects_min`, `num_objects_max` — how many PLYs to sample
- `position_range` — bounding box for random translation
- `rotation_range` — max random Euler angles (degrees)
- `scale_jitter_range` — uniform scale multiplier range
- `min_separation` (optional v1.1) — reject placements closer than threshold

Steps:

1. Sample N PLY paths (with replacement if fewer PLYs than N)
2. Load each, apply random transform
3. Concatenate into one `SceneGaussians` with distinct `object_ids`
4. Return metadata: list of `{object_id, ply_path, transform}`

### 6.3 `camera.py`

**`sample_random_camera(scene_bounds, config) -> viewmat, K, width, height`**

Config parameters:

- `image_width`, `image_height` (default **512×512** for CPU feasibility)
- `fov_deg` range
- `camera_distance_range` — distance from scene center
- `look_at_jitter` — small random offset from scene centroid

Use pinhole camera:

- `K` = 3×3 intrinsics from FOV + image size
- `viewmat` = 4×4 world-to-camera (OpenGL or COLMAP convention — pick one, document it, use consistently)

Ensure all objects are in frustum; reject/resample camera if scene projects outside image or is empty.

### 6.4 `render/rasterizer.py`

**Primary deliverable.** Adapt tile-based CPU rasterizer from gsplat-pytorch or EasyGaussianSplatting.

**Public API:**

```python
def render(
    gaussians: SceneGaussians,
    viewmat: Tensor,       # [4, 4]
    K: Tensor,             # [3, 3]
    width: int,
    height: int,
) -> RenderOutput:

@dataclass
class RenderOutput:
    fg_rgb: Tensor        # [H, W, 3] float 0-1, splats over black
    alpha: Tensor         # [H, W]    float 0-1, splat coverage
    object_id_map: Tensor # [H, W]    int, -1 = no splat
```

Implementation notes:

- Use **tile-based** rasterization (e.g. 16×16 tiles) for reasonable CPU speed
- Sort Gaussians by depth per tile (or globally for v1 simplicity)
- Evaluate SH degree-0 only in v1 (`color = sh_dc`); higher SH can be v1.1
- Cull Gaussians with zero radius or behind camera
- Set `torch.set_num_threads(n)` per worker to avoid oversubscription in multiprocessing
- Do **not** composite background inside this module

### 6.5 `background.py`

```python
def composite(
    fg_rgb: Tensor,
    alpha: Tensor,
    background: BackgroundSpec,
    width: int,
    height: int,
) -> Tensor:
    """Composite foreground splats over solid color or image.

    v1: BackgroundSpec.mode == "solid" only.
    v2: BackgroundSpec.mode == "image" with resize/crop.
    """
```

v1 implements solid color. Include a docstring describing the v2 image path.

### 6.6 `picker.py`

```python
def sample_click(
    alpha: Tensor,
    object_id_map: Tensor,
    alpha_threshold: float = 0.5,
    rng: np.random.Generator,
) -> tuple[int, int, int]:  # x, y, object_id
```

- Collect all `(x, y)` where `alpha > threshold` and `object_id >= 0`
- Uniformly sample one
- Return `(x, y, object_id_map[y, x])`

### 6.7 `sample.py`

```python
def generate_one_sample(ply_paths, config, rng, output_dir) -> SampleRecord
```

Pipeline:

1. `build_random_scene(...)`
2. `sample_random_camera(...)` (retry up to `max_camera_retries`)
3. `render(...)` → `fg_rgb`, `alpha`, `object_id_map`
4. `composite(...)` → `rgb`
5. `sample_click(...)`
6. `mask = (object_id_map == clicked_object_id) * 255`
7. Export via `export.py`
8. Return metadata record

### 6.8 `export.py`

Per sample, write:

```
outputs/run_001/
├── images/000001.png          # composited RGB
├── masks/000001.png           # binary mask (mode L, 0 or 255)
└── annotations.jsonl          # one JSON object per line, appended
```

**JSONL record schema:**

```json
{
  "id": "000001",
  "image": "images/000001.png",
  "mask": "masks/000001.png",
  "point": [x, y],
  "object_id": 2,
  "num_objects": 4,
  "background": {
    "mode": "solid",
    "color": [0.1, 0.1, 0.1]
  },
  "camera": {
    "width": 512,
    "height": 512,
    "fov_deg": 60.0,
    "viewmat": [[...4x4...]],
    "K": [[...3x3...]]
  },
  "objects": [
    {"object_id": 0, "ply": "chair.ply"},
    {"object_id": 1, "ply": "mug.ply"}
  ]
}
```

Also write `outputs/run_001/config.yaml` snapshot at start of run.

### 6.9 `scripts/generate_dataset.py`

Typer CLI:

```bash
python scripts/generate_dataset.py \
  --ply-dir assets/ply \
  --output outputs/run_001 \
  --num-samples 100 \
  --workers 4 \
  --config configs/default.yaml \
  --seed 42
```

Flags:

- `--ply-dir` — directory of `.ply` files
- `--output` — output directory
- `--num-samples` — number of samples to generate
- `--workers` — multiprocessing worker count (default: CPU count - 1)
- `--config` — YAML config path
- `--seed` — master RNG seed

Each worker gets `(seed + worker_id)` for reproducibility.

---

## 7. Default Config (`configs/default.yaml`)

```yaml
render:
  width: 512
  height: 512
  alpha_threshold: 0.5
  sh_degree: 0              # v1: DC only

background:
  mode: "solid"             # v1: "solid" only; v2: "image"
  solid_color: [0.1, 0.1, 0.1]
  # v2 (not implemented in v1):
  # image_dir: "assets/backgrounds/"
  # resize_mode: "crop"     # crop | letterbox | stretch

scene:
  num_objects_min: 2
  num_objects_max: 5
  position_range: [-2.0, 2.0]   # per axis
  rotation_deg_max: 180
  scale_jitter: [0.8, 1.2]

camera:
  fov_deg_range: [45, 75]
  distance_range: [3.0, 8.0]
  max_retries: 20

generation:
  max_camera_retries: 20
```

---

## 8. Implementation Phases

Implement in this order. Complete each phase before moving on.

### Phase 1 — Scaffold + PLY loader

- [ ] Create `pyproject.toml` with dependencies
- [ ] Implement `ply_loader.py`
- [ ] Test loading a real 3DGS PLY; print Gaussian count and bounds
- [ ] Add `tests/test_ply_loader.py`

### Phase 2 — Camera + single-object render

- [ ] Implement `camera.py`
- [ ] Vendor/adapt CPU rasterizer into `render/`
- [ ] Implement `background.py` (solid color compositing)
- [ ] Render **one** PLY from one camera → save debug PNG
- [ ] Verify render looks reasonable (not blank, not noise)

### Phase 3 — Multi-object scenes

- [ ] Implement `scene.py` (concatenate transformed objects)
- [ ] Render multi-object RGB
- [ ] Verify objects appear at different positions

### Phase 4 — Object ID map + occlusion-aware mask

- [ ] Extend rasterizer to output `object_id_map` using dominant-weight rule (§3.2)
- [ ] Implement `picker.py`
- [ ] Implement mask = `(object_id_map == clicked_id)`
- [ ] **`tests/test_mask_occlusion.py`**: two objects, one in front of other; assert mask of rear object is black where front object occludes it
- [ ] Assert `mask[y, x] == 255` at click pixel

### Phase 5 — Dataset export + CLI

- [ ] Implement `export.py`, `sample.py`, `parallel.py`
- [ ] Implement `scripts/generate_dataset.py`
- [ ] Generate 10-sample smoke test dataset
- [ ] Validate JSONL schema

### Phase 6 — README + polish

- [ ] README with setup, usage, expected performance notes
- [ ] `.gitignore` (`outputs/`, `__pycache__/`, `.venv/`)
- [ ] Document v2 background image path in README (not implemented)

---

## 9. Testing Requirements

### Unit tests

| Test | Asserts |
|---|---|
| `test_ply_loader` | Loads PLY, correct tensor shapes, opacities in (0,1) |
| `test_camera` | View matrix is valid rotation; K matches FOV |
| `test_single_object` | Single object → mask equals foreground splat regions |
| `test_mask_occlusion` | Two stacked objects; rear object mask has black occluded region |

### Occlusion test setup (critical)

Construct synthetic scene with **two identical small splat objects**:

- Object 0 at `z = 0`
- Object 1 at same `(x, y)` but closer to camera (smaller `z` or closer along view ray)
- Click a pixel where both overlap
- Clicked object = front object → mask is white blob
- Clicked object = rear object → mask is **only the visible ring**, black where front object covers it

If no real PLYs available for CI, use **synthetic Gaussians** (a few Gaussians per object with known positions) in the test.

---

## 10. Performance Expectations (CPU)

Be explicit in README:

| Setting | Rough expectation |
|---|---|
| 512×512, 50k Gaussians/scene, 1 worker | ~5–30 s/sample |
| 4 workers | ~4× throughput (not perfect due to memory) |
| 256×256 | ~4× faster |

Mitigations:

- Default 512×512; allow `--width 256` for dev iteration
- Limit objects per scene (2–5)
- Use multiprocessing
- Cap Gaussians per PLY if needed (`max_gaussians_per_object` config, optional v1.1)

Do **not** block v1 on real-time performance. Correctness first.

---

## 11. Acceptance Criteria (v1 done when)

1. CLI generates N samples from a directory of PLY files on **CPU-only** machine (no CUDA)
2. Each sample has RGB PNG, mask PNG, and JSONL record with `(x, y)` point
3. Mask is **object-level** and **occlusion-aware** (covered parts black)
4. Click pixel is always inside the white region of its mask
5. Occlusion unit test passes
6. README documents setup and usage
7. `RenderOutput` includes separate `fg_rgb` and `alpha`; compositing happens in `background.py`
8. Config has a `background` section with `mode: "solid"` (image mode documented but unimplemented)
9. `background.py` exists with a solid-color compositor and a docstring describing the v2 image path

---

## 12. Out of Scope (v1)

- GPU / CUDA rendering
- Higher-order SH (degree > 0) — optional v1.1
- Background **image** compositing (solid color only in v1; architecture must support images in v2)
- Collision detection between objects
- Amodal masks (full object including hidden parts)
- Model training code
- Web viewer

---

## 13. Dependencies (`pyproject.toml`)

```toml
[project]
name = "splat-dataset"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch",           # CPU wheel
    "numpy",
    "plyfile",
    "pillow",
    "typer",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
generate-dataset = "splat_dataset.cli:app"   # optional; scripts/ also fine
```

---

## 14. Key Design Decisions (do not change without discussion)

| Decision | Choice |
|---|---|
| Click target | Object (PLY instance), not individual Gaussian |
| Mask type | **Visible** (modal) — occluded parts black |
| Mask method | Single-pass `object_id_map`, not per-object re-render |
| Renderer | Pure PyTorch CPU, vendored rasterizer |
| Background | Composited in `background.py` after splat pass; v1 solid, v2 images |
| Coordinate convention | Pixel `(x, y)` = column, row; origin top-left |
| Point storage | Integer pixel coords in JSONL |

---

## 15. Composer Implementation Notes

- **Vendoring:** Copy/adapt rasterizer code from gsplat-pytorch or EasyGaussianSplatting into `src/splat_dataset/render/`. Add source attribution comment at top of vendored files.
- **SH v1:** Using DC-only (`sh_dc`) colors is acceptable for v1. Document in README.
- **Convention:** Use COLMAP camera convention if EasyGaussianSplatting code is adapted; document in `camera.py`.
- **Do not** introduce GPU code paths in v1.
- **Do not** use SAM or any ML model for labels.
- Prioritize **`test_mask_occlusion`** — it validates the core requirement.
- If rasterizer is too slow even for smoke tests, allow `width=128, height=128` in tests only.
- **Background v1:** Implement solid color only, but split rendering into `fg_rgb` + `alpha` and composite via `background.py` so random background images can be added in v2 without changing the rasterizer, mask logic, or click sampling.
