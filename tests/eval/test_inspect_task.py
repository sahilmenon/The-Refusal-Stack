"""Smoke test for the Inspect AI task builder - no real model or HF downloads."""

from __future__ import annotations

from unittest.mock import MagicMock

import datasets as hf_datasets
import pytest

pytest.importorskip("inspect_ai", reason="inspect_ai not installed in this environment")


def _tiny_dataset() -> hf_datasets.Dataset:
    return hf_datasets.Dataset.from_dict(
        {
            "prompt": [f"harmful thing {i}" for i in range(5)],
            "label": ["harmful"] * 5,
        }
    )


def test_build_inspect_dataset():
    from refusal_stack.eval.inspect_task import build_inspect_dataset

    samples = build_inspect_dataset(_tiny_dataset(), "advbench")
    assert len(samples) == 5
    for s in samples:
        assert s.target == "harmful"
        assert "dataset" in s.metadata


def test_build_refusal_task_returns_task():
    from inspect_ai import Task

    from refusal_stack.eval.config import EvalConfig
    from refusal_stack.eval.inspect_task import build_refusal_task

    wrapper = MagicMock()
    wrapper.model_id = "test/model"
    wrapper.revision = None
    config = EvalConfig(model_id="test/model", datasets=["advbench"], held_out_seed=42)
    task = build_refusal_task(_tiny_dataset(), "advbench", wrapper, None, config, None)
    assert isinstance(task, Task)
