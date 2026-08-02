"""Minimal .env loader (no dependency).

The cloud tooling reads credentials from ``os.environ``. This loads a local
``.env`` into the environment so ``make preflight`` / ``make pod-*`` work without
a manual ``export``. Existing environment values are never overwritten, and the
file is optional (missing .env is a no-op).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dotenv(path: str = ".env") -> int:
    """Load KEY=VALUE lines from ``path`` into os.environ. Returns keys loaded."""
    p = Path(path)
    if not p.exists():
        return 0
    loaded = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        logger.debug("Loaded %d var(s) from %s", loaded, path)
    return loaded
