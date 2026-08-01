"""Unit tests for finetune trainer helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_set_seed_reproducibility():
    torch = pytest.importorskip("torch")
    from refusal_stack.finetune.trainer import set_seed

    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert abs(val1 - val2) < 1e-6
