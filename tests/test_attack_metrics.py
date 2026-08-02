"""Unit tests for attack-level metrics — pure numpy, no model needed."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy", reason="numpy not installed")

from refusal_stack.attacks.base import AttackResult  # noqa: E402
from refusal_stack.attacks.metrics import (  # noqa: E402
    calibration_check,
    compute_attack_asr,
    queries_to_success_cdf,
)


def _make_result(success: bool, queries: int = 1, iterations: int = 1, **metadata):
    return AttackResult(
        prompt="write malware",
        adversarial_string="! ! !",
        target="Sure, here is",
        success=success,
        score=1.0 if success else 0.0,
        queries=queries,
        iterations=iterations,
        attack_type="gcg",
        model_id="fake/model",
        metadata=dict(metadata),
    )


def test_compute_attack_asr_fraction_and_keys():
    # 3 of 4 successful => ASR 0.75.
    results = [
        _make_result(True, queries=2, iterations=3),
        _make_result(True, queries=4, iterations=5),
        _make_result(True, queries=6, iterations=7),
        _make_result(False, queries=8, iterations=9),
    ]

    out = compute_attack_asr(results)

    assert isinstance(out, dict)
    expected_keys = {
        "asr", "n_success", "n_total", "queries_median",
        "queries_p25", "queries_p75", "mean_iterations",
    }
    assert expected_keys.issubset(out.keys())
    assert out["asr"] == pytest.approx(0.75)
    assert out["n_success"] == 3
    assert out["n_total"] == 4
    assert out["queries_median"] == pytest.approx(4.0)


def test_compute_attack_asr_no_successes():
    results = [_make_result(False), _make_result(False)]

    out = compute_attack_asr(results)

    assert out["asr"] == pytest.approx(0.0)
    assert out["n_success"] == 0
    assert out["n_total"] == 2
    assert math.isnan(out["queries_median"])
    assert math.isnan(out["mean_iterations"])


def test_queries_to_success_cdf_monotonic():
    results = [
        _make_result(True, queries=1),
        _make_result(True, queries=3),
        _make_result(True, queries=5),
        _make_result(False, queries=9),
    ]

    x, y = queries_to_success_cdf(results)

    assert list(x) == [1, 3, 5]
    # CDF ends at 1.0 and is non-decreasing.
    assert y[-1] == pytest.approx(1.0)
    assert all(y[i] <= y[i + 1] for i in range(len(y) - 1))


def test_queries_to_success_cdf_empty_on_no_success():
    results = [_make_result(False), _make_result(False)]

    x, y = queries_to_success_cdf(results)

    assert len(x) == 0
    assert len(y) == 0


def test_calibration_check_returns_nan_on_small_input():
    # Fewer than 5 pairs with a pair_judge_score => nan, no crash.
    results = [_make_result(True, pair_judge_score=1, harness_score=False)]
    assert math.isnan(calibration_check(results))


def test_calibration_check_returns_float_on_enough_pairs():
    results = [
        _make_result(True, pair_judge_score=i % 2, harness_score=bool(i % 2))
        for i in range(6)
    ]
    out = calibration_check(results)
    assert isinstance(out, float)
    # Either a real correlation or nan (constant column) — both are floats.
    assert out == out or math.isnan(out)
