"""Tests for 3DGS PLY loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ply_loader import SceneGaussians, load_ply, write_synthetic_ply


def test_load_synthetic_ply(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=8)
    scene, _stats = load_ply(ply_path)

    assert isinstance(scene, SceneGaussians)
    assert scene.num_gaussians == 8
    assert scene.means.shape == (8, 3)
    assert scene.quats.shape == (8, 4)
    assert scene.scales.shape == (8, 3)
    assert scene.opacities.shape == (8,)
    assert scene.sh_dc.shape == (8, 3)
    assert scene.sh_rest.shape == (8, 0, 3)
    assert scene.object_ids.shape == (8,)
    assert np.all(scene.object_ids.numpy() == 0)


def test_activations_in_valid_ranges(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=4)
    scene, _stats = load_ply(ply_path)

    opacities = scene.opacities.numpy()
    assert np.all(opacities > 0.0)
    assert np.all(opacities < 1.0)

    scales = scene.scales.numpy()
    assert np.all(scales > 0.0)

    quat_norms = np.linalg.norm(scene.quats.numpy(), axis=-1)
    np.testing.assert_allclose(quat_norms, 1.0, rtol=1e-5, atol=1e-5)


def test_bounds(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=16)
    scene, _stats = load_ply(ply_path)
    lo, hi = scene.bounds()
    assert lo.shape == (3,)
    assert hi.shape == (3,)
    assert np.all(lo <= hi)


def test_normalize_to_unit_extent(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=64)
    scene, stats = load_ply(ply_path)
    lo, hi = scene.bounds()
    extent = hi - lo
    center = (lo + hi) / 2.0

    assert stats.max_extent_before > 0.0
    assert stats.max_extent_after == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert extent.max() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(center, 0.0, atol=1e-5)


def test_missing_fields_raise(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.ply"
    bad_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
    )
    with pytest.raises(ValueError, match="missing fields"):
        load_ply(bad_path)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_ply("/nonexistent/object.ply")


def test_max_gaussians_random_subsample(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=32)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    rng_c = np.random.default_rng(1)

    sub_a, _ = load_ply(ply_path, max_gaussians=8, rng=rng_a)
    sub_b, _ = load_ply(ply_path, max_gaussians=8, rng=rng_b)
    sub_c, _ = load_ply(ply_path, max_gaussians=8, rng=rng_c)

    assert sub_a.num_gaussians == 8
    np.testing.assert_allclose(sub_a.means.numpy(), sub_b.means.numpy())
    assert not np.allclose(sub_a.means.numpy(), sub_c.means.numpy())


def test_draw_fraction_range_subsample(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=100)
    rng = np.random.default_rng(0)

    scene, stats = load_ply(
        ply_path,
        max_gaussians=50,
        draw_fraction_range=[0.5, 0.5],
        rng=rng,
    )

    assert scene.num_gaussians == 25
    assert stats.vertex_count == 100
    assert stats.draw_base == 50
    assert stats.draw_fraction == pytest.approx(0.5)


def test_f_rest_channel_major_layout(tmp_path: Path) -> None:
    """Match Inria 3DGS PLY layout: f_rest_0..14=R, 15..29=G, 30..44=B."""
    from plyfile import PlyData, PlyElement

    n = 1
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ] + [(f"f_rest_{i}", "f4") for i in range(45)]
    row = np.zeros((), dtype=dtype)
    row["x"], row["y"], row["z"] = 0.0, 0.0, 0.0
    row["opacity"] = 0.0
    row["scale_0"] = row["scale_1"] = row["scale_2"] = np.log(0.05)
    row["rot_0"] = 1.0
    row["f_dc_0"] = row["f_dc_1"] = row["f_dc_2"] = 0.0
    row["f_rest_0"] = 1.25
    row["f_rest_15"] = 2.5
    row["f_rest_30"] = 3.75
    rows = np.array([row] * n)
    ply_path = tmp_path / "sh_layout.ply"
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(ply_path))

    scene, _stats = load_ply(ply_path)
    sh = scene.sh_rest.numpy()[0]

    assert sh.shape == (15, 3)
    np.testing.assert_allclose(sh[0], [1.25, 2.5, 3.75], rtol=0, atol=1e-6)
    assert np.count_nonzero(sh) == 3


def test_draw_fraction_uses_full_ply_without_cap(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=20)
    rng = np.random.default_rng(0)

    scene, stats = load_ply(
        ply_path,
        draw_fraction_range=[0.25, 0.25],
        rng=rng,
    )

    assert scene.num_gaussians == 5
    assert stats.draw_base == 20
    assert stats.draw_fraction == pytest.approx(0.25)
