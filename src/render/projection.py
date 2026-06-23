# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# 3D Gaussian projection and 2D covariance computation.

from __future__ import annotations

import torch

from camera import project_points, transform_world_to_camera


def _upper_triangular_3d(mat: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            mat[:, 0, 0],
            mat[:, 0, 1],
            mat[:, 0, 2],
            mat[:, 1, 1],
            mat[:, 1, 2],
            mat[:, 2, 2],
        ],
        dim=-1,
    )


def _symmetric_from_upper_3d(upper: torch.Tensor) -> torch.Tensor:
    n = upper.shape[0]
    mat = torch.zeros(n, 3, 3, dtype=upper.dtype, device=upper.device)
    mat[:, 0, 0] = upper[:, 0]
    mat[:, 0, 1] = mat[:, 1, 0] = upper[:, 1]
    mat[:, 0, 2] = mat[:, 2, 0] = upper[:, 2]
    mat[:, 1, 1] = upper[:, 3]
    mat[:, 1, 2] = mat[:, 2, 1] = upper[:, 4]
    mat[:, 2, 2] = upper[:, 5]
    return mat


def _upper_triangular_2d(mat: torch.Tensor) -> torch.Tensor:
    return torch.stack([mat[:, 0, 0], mat[:, 0, 1], mat[:, 1, 1]], dim=-1)


def compute_cov_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    n = scales.shape[0]
    device = scales.device
    dtype = scales.dtype

    s = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    s[:, 0, 0] = scales[:, 0]
    s[:, 1, 1] = scales[:, 1]
    s[:, 2, 2] = scales[:, 2]

    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    r = torch.stack(
        [
            torch.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )

    m = r @ s
    sigma = m @ m.transpose(1, 2)
    return _upper_triangular_3d(sigma)


def compute_cov_2d(
    points_cam: torch.Tensor,
    cov3d_upper: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    fx, fy = k[0, 0], k[1, 1]
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    tan_fovx = 2.0 * torch.atan(width / (2.0 * fx))
    tan_fovy = 2.0 * torch.atan(height / (2.0 * fy))
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy

    z_safe = z.clamp(min=1e-4)
    x = (x / z_safe).clamp(-limx, limx) * z
    y = (y / z_safe).clamp(-limy, limy) * z

    z2 = z * z
    j = torch.zeros(points_cam.shape[0], 3, 3, device=points_cam.device, dtype=points_cam.dtype)
    j[:, 0, 0] = fx / z
    j[:, 0, 2] = -(fx * x) / z2
    j[:, 1, 1] = fy / z
    j[:, 1, 2] = -(fy * y) / z2

    rcw = viewmat[:3, :3]
    t = j @ rcw

    sigma = _symmetric_from_upper_3d(cov3d_upper)
    sigma_prime = t @ sigma @ t.transpose(1, 2)
    sigma_prime[:, 0, 0] += 0.3
    sigma_prime[:, 1, 1] += 0.3

    return _upper_triangular_2d(sigma_prime[:, :2, :2])


def inverse_cov2d(cov2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    det = cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] * cov2d[:, 1] + 1e-6
    det_inv = 1.0 / det
    cinv = torch.stack(
        [
            cov2d[:, 2] * det_inv,
            -cov2d[:, 1] * det_inv,
            cov2d[:, 0] * det_inv,
        ],
        dim=-1,
    )
    areas = (3.0 * torch.sqrt(torch.stack([cov2d[:, 0], cov2d[:, 2]], dim=-1))).to(torch.int32)
    return cinv, areas


def project_gaussians(
    means: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_cam = transform_world_to_camera(means, viewmat)
    uv, depth = project_points(points_cam, k)
    return uv, points_cam, depth
