"""Shared pytest fixtures for the refusal-stack test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import datasets as hf_datasets
import pytest


@pytest.fixture()
def tiny_harmful_dataset():
    return hf_datasets.Dataset.from_dict(
        {
            "prompt": [f"Describe how to do harmful thing {i}" for i in range(10)],
            "label": ["harmful"] * 10,
        }
    )


@pytest.fixture()
def tiny_benign_dataset():
    return hf_datasets.Dataset.from_dict(
        {
            "prompt": [f"Explain topic {i}" for i in range(10)],
            "label": ["benign"] * 10,
        }
    )


@pytest.fixture()
def mock_wrapper():
    wrapper = MagicMock()
    wrapper.model_id = "test/model"
    wrapper.revision = None

    def _generate_batch(prompts, seed=42):
        return ["I can't help with that."] * len(prompts)

    wrapper.generate_batch.side_effect = _generate_batch
    return wrapper


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (require real credentials and GPU)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="Pass --integration to run integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
