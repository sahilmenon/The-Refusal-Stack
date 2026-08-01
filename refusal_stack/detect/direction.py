"""Load refusal direction artifact from Phase 3."""
from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_refusal_direction(path: str) -> tuple[torch.Tensor, int]:
    weights = load_file(path)
    direction = weights["direction"]
    assert direction.ndim == 1, f"Expected 1-D direction, got shape {direction.shape}"

    with safe_open(path, framework="pt") as f:
        meta = f.metadata()

    layer_idx = int(meta["layer_idx"])
    return direction.to(torch.float32), layer_idx
