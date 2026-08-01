"""Shared utilities for the eval pipeline."""

from __future__ import annotations

import logging
import os
import random


class ChatTemplateError(Exception):
    """Raised when a tokenizer lacks a chat template but one is required."""


def set_all_seeds(seed: int) -> None:
    """Fix every RNG source that affects model output or data ordering.

    Setting PYTHONHASHSEED only takes effect for *new* interpreter processes, but
    we set it anyway so that subprocesses spawned from this one inherit a stable
    hash seed rather than a random one.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # numpy is an optional peer-dep in some lightweight environments, but we treat it
    # as present because the broader eval pipeline needs it unconditionally.
    import numpy as np  # noqa: PLC0415

    np.random.seed(seed)

    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN kernels trade a little speed for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a human-readable format.

    Using basicConfig is intentional: downstream callers that want a different
    handler (e.g. Rich) can replace the root handlers after this call.  We avoid
    forcing a specific handler here so that library users retain control.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
