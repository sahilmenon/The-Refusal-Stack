"""Generation cache backed by a JSON-lines file.

Caching avoids re-running inference on prompts already scored, which matters
for iterative development and for the --cache-only CLI flag that recomputes
metrics without touching the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

try:
    import fcntl as _fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class GenerationCache:
    """Persistent cache keyed by SHA256(model_id + revision + prompt)."""

    def __init__(self, cache_dir: str, model_id: str, dataset_name: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        slug = model_id.replace("/", "_")
        self._path = self._dir / f"{slug}_{dataset_name}.jsonl"
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["generation"]
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed cache line in %s", self._path)

    @staticmethod
    def make_key(model_id: str, revision: str | None, prompt: str) -> str:
        payload = f"{model_id}|{revision or ''}|{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, generation: str) -> None:
        if key in self._cache:
            return
        self._cache[key] = generation
        entry = json.dumps({"key": key, "generation": generation}, ensure_ascii=False)
        with _lock:
            with self._path.open("a", encoding="utf-8") as fh:
                if _HAS_FCNTL:
                    try:
                        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                        fh.write(entry + "\n")
                    finally:
                        try:
                            _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                        except Exception:
                            pass
                else:
                    fh.write(entry + "\n")
