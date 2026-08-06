from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


_FINGERPRINT_FILE = "fingerprint.txt"


class ActivationCacheWriter:
    def __init__(self, cache_dir: str, run_id: str):
        self._dir = Path(cache_dir) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_layer(self, layer_idx: int, label: str, activations: np.ndarray) -> None:
        path = self._dir / f"layer_{layer_idx:02d}_{label}.npy"
        np.save(str(path), activations)

    def write_fingerprint(self, fingerprint: str) -> None:
        """Record the config fingerprint this cache was built under, so a later
        run with different inputs (model / counts / seed / split) invalidates it
        instead of silently loading stale activations."""
        (self._dir / _FINGERPRINT_FILE).write_text(fingerprint, encoding="utf-8")


class ActivationCacheReader:
    def __init__(self, cache_dir: str, run_id: str):
        self._dir = Path(cache_dir) / run_id

    def load_layer(self, layer_idx: int, label: str) -> np.ndarray:
        path = self._dir / f"layer_{layer_idx:02d}_{label}.npy"
        return np.load(str(path))

    def exists(self, layer_idx: int, label: str) -> bool:
        return (self._dir / f"layer_{layer_idx:02d}_{label}.npy").exists()

    def fingerprint(self) -> str | None:
        """The config fingerprint this cache was built under, or None for a
        legacy cache written before fingerprinting existed."""
        path = self._dir / _FINGERPRINT_FILE
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def is_fresh_for(self, fingerprint: str) -> bool:
        """True only if cached activations were built under ``fingerprint``. A
        legacy cache with no recorded fingerprint is treated as stale, so a
        resumed run can never silently reuse activations from a different config."""
        stored = self.fingerprint()
        return stored is not None and stored == fingerprint
