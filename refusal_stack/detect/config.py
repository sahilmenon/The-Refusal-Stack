"""Pydantic config for the tamper detector."""

from __future__ import annotations

import yaml
from pydantic import BaseModel


class DetectConfig(BaseModel):
    refusal_direction_path: str = "artifacts/refusal_direction_latest.safetensors"
    layer_idx: int | None = None
    batch_size: int = 8
    device: str = "cuda"
    threshold_fpr_target: float = 0.05


def load_detect_config(path: str) -> DetectConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DetectConfig(**raw)
