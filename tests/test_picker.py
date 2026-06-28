"""Click sampling unit tests."""

from __future__ import annotations

import numpy as np
import torch

from picker import sample_click


def test_sample_click_with_object_weights_int32_id_map() -> None:
    """Rasterizer emits int32 object_id_map; gather indices must be int64."""
    alpha = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    object_id_map = torch.tensor([[-1, 0], [1, -1]], dtype=torch.int32)
    object_weights = torch.tensor(
        [
            [[0.0, 0.0], [0.9, 0.0]],
            [[0.0, 0.8], [0.0, 0.0]],
        ]
    )
    rng = np.random.default_rng(0)

    x, y, oid = sample_click(
        alpha,
        object_id_map,
        0.5,
        rng,
        object_weights=object_weights,
        weight_threshold=0.05,
    )

    assert oid in {0, 1}
    assert alpha[y, x].item() > 0.5
