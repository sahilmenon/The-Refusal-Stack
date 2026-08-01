from __future__ import annotations
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActivationCacheWriter:
    def __init__(self, cache_dir: str, run_id: str):
        self._dir = Path(cache_dir) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_layer(self, layer_idx: int, label: str, activations: np.ndarray) -> None:
        path = self._dir / f"layer_{layer_idx:02d}_{label}.npy"
        np.save(str(path), activations)


class ActivationCacheReader:
    def __init__(self, cache_dir: str, run_id: str):
        self._dir = Path(cache_dir) / run_id

    def load_layer(self, layer_idx: int, label: str) -> np.ndarray:
        path = self._dir / f"layer_{layer_idx:02d}_{label}.npy"
        return np.load(str(path))

    def exists(self, layer_idx: int, label: str) -> bool:
        return (self._dir / f"layer_{layer_idx:02d}_{label}.npy").exists()
