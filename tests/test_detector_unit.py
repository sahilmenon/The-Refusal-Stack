"""Unit tests for the Phase 4 tamper detector."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def base_proj():
    rng = np.random.default_rng(0)
    return rng.normal(2.0, 0.5, size=50)


@pytest.fixture()
def test_proj():
    rng = np.random.default_rng(1)
    return rng.normal(-2.0, 0.5, size=50)


def test_compute_auroc_perfect_separation(base_proj, test_proj):
    from refusal_stack.detect.scorer import TamperDetector

    detector = TamperDetector(base_proj, test_proj)
    auroc = detector.compute_auroc()
    assert auroc >= 0.95


def test_compute_auroc_random():
    from refusal_stack.detect.scorer import TamperDetector

    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 100)
    detector = TamperDetector(a, b)
    auroc = detector.compute_auroc()
    assert 0.35 <= auroc <= 0.65


def test_threshold_fpr(base_proj, test_proj):
    from refusal_stack.detect.scorer import TamperDetector

    detector = TamperDetector(base_proj, test_proj)
    threshold = detector.fit_threshold(fpr_target=0.05)
    assert isinstance(threshold, float)
    classified_base = detector.classify(base_proj)
    empirical_fpr = float(np.mean(classified_base))
    assert empirical_fpr <= 0.15


def test_classify_output_shape(base_proj, test_proj):
    from refusal_stack.detect.scorer import TamperDetector

    detector = TamperDetector(base_proj, test_proj)
    detector.fit_threshold(0.05)
    result = detector.classify(np.zeros(20))
    assert result.shape == (20,)
    assert result.dtype == bool


def test_separation_dict(base_proj, test_proj):
    from refusal_stack.detect.scorer import TamperDetector

    detector = TamperDetector(base_proj, test_proj)
    sep = detector.compute_separation()
    assert "base_mean" in sep
    assert "test_mean" in sep
    assert "cohen_d" in sep
    assert sep["base_mean"] > sep["test_mean"]


def test_load_refusal_direction_shape(tmp_path):
    pytest.importorskip("safetensors", reason="safetensors not installed")
    import torch
    from safetensors.torch import save_file

    from refusal_stack.detect.direction import load_refusal_direction

    direction = torch.randn(4096)
    path = str(tmp_path / "dir.safetensors")
    save_file(
        {"direction": direction},
        path,
        metadata={"layer_idx": "15", "norm": "1.0", "model_id": "test"},
    )
    loaded, layer_idx = load_refusal_direction(path)
    assert loaded.shape == (4096,)
    assert layer_idx == 15
