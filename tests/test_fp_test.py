"""Tests for false-positive test."""
from __future__ import annotations

import numpy as np


def test_fp_test_perfect_separation():
    from refusal_stack.detect.fp_test import run_false_positive_test
    base = np.ones(50) * 2.0
    benign = np.ones(50) * 1.5
    # threshold well below the benign distribution
    result = run_false_positive_test(base, benign, threshold=-100.0)
    assert result["fpr"] == 0.0
    assert result["pass"] is True


def test_fp_test_identical_distributions():
    from refusal_stack.detect.fp_test import run_false_positive_test
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 100)
    benign = rng.normal(0, 1, 100)
    # Set threshold to classify all as tampered
    result = run_false_positive_test(base, benign, threshold=1e6)
    assert result["fpr"] > 0.5
    assert result["pass"] is False
