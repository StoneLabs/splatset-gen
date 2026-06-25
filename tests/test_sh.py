import pytest
import torch

from render.sh import MAX_SH_DEGREE, SH_C0, eval_sh, eval_sh_dc, eval_sh_view, validate_sh_degree


def test_validate_sh_degree_rejects_out_of_range():
    validate_sh_degree(0)
    validate_sh_degree(MAX_SH_DEGREE)
    with pytest.raises(ValueError, match="render.sh_degree must be 0–3"):
        validate_sh_degree(4)
    with pytest.raises(ValueError, match="render.sh_degree must be 0–3"):
        validate_sh_degree(-1)


def test_eval_sh_degree0_matches_dc():
    sh_dc = torch.tensor([[0.2, -0.1, 0.4], [0.0, 0.5, -0.3]])
    sh = sh_dc.unsqueeze(-1)
    dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(eval_sh(0, sh, dirs), SH_C0 * sh_dc)
    assert torch.allclose(eval_sh_dc(sh_dc), eval_sh_view(sh_dc, torch.zeros(2, 0, 3), torch.zeros(2, 3), torch.zeros(3), 0))


def test_eval_sh_view_uses_higher_order_coeffs():
    sh_dc = torch.zeros(1, 3)
    sh_rest = torch.zeros(1, 3, 3)
    sh_rest[0, 1, 0] = 0.5  # z-axis SH coeff, R channel

    means = torch.tensor([[0.0, 0.0, 0.0]])
    cam_a = torch.tensor([0.0, 0.0, 1.0])
    cam_b = torch.tensor([1.0, 0.0, 0.0])

    color_a = eval_sh_view(sh_dc, sh_rest, means, cam_a, sh_degree=1)
    color_b = eval_sh_view(sh_dc, sh_rest, means, cam_b, sh_degree=1)
    assert not torch.allclose(color_a, color_b)


def test_effective_degree_caps_to_available_rest():
    sh_dc = torch.zeros(1, 3)
    sh_rest = torch.zeros(1, 3, 3)
    sh_rest[0, 1, 0] = 0.5  # z-axis SH coeff, R channel

    means = torch.tensor([[0.0, 0.0, 0.0]])
    cam = torch.tensor([0.0, 0.0, 1.0])

    full = eval_sh_view(sh_dc, sh_rest, means, cam, sh_degree=1)
    capped = eval_sh_view(sh_dc, sh_rest, means, cam, sh_degree=3)
    assert torch.allclose(full, capped)
