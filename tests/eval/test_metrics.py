"""Tests for eval metrics and the results dataframe."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from refusal_stack.eval.scorers import RefusalScore
from refusal_stack.eval.metrics import (
    build_results_dataframe,
    compute_asr,
    compute_false_refusal_rate,
    compute_refusal_rate,
    compute_summary_stats,
)


def _make_score(is_refusal: bool, partial: bool = False) -> RefusalScore:
    return RefusalScore(
        is_refusal=is_refusal,
        partial_compliance=partial,
        regex_hit=is_refusal,
        raw_text="test",
    )


def test_refusal_rate_all_refusals():
    scores = [_make_score(True)] * 5
    labels = ["harmful"] * 5
    assert compute_refusal_rate(scores, labels) == pytest.approx(1.0)


def test_refusal_rate_all_compliant():
    scores = [_make_score(False)] * 5
    labels = ["harmful"] * 5
    assert compute_refusal_rate(scores, labels) == pytest.approx(0.0)


def test_refusal_rate_mixed():
    scores = [_make_score(True), _make_score(False), _make_score(True), _make_score(False)]
    labels = ["harmful"] * 4
    assert compute_refusal_rate(scores, labels) == pytest.approx(0.5)


def test_false_refusal_rate():
    scores = [_make_score(True), _make_score(False), _make_score(False)]
    labels = ["benign"] * 3
    result = compute_false_refusal_rate(scores, labels)
    assert result == pytest.approx(1 / 3)


def test_asr():
    scores = [_make_score(True)] * 3 + [_make_score(False)] * 7
    labels = ["harmful"] * 10
    asr = compute_asr(scores, labels)
    assert asr == pytest.approx(0.7)


def test_build_results_dataframe_columns():
    scores = [_make_score(True), _make_score(False)]
    labels = ["harmful", "benign"]
    prompts = ["prompt 1", "prompt 2"]
    generations = ["I can't help", "Sure!"]
    dataset_names = ["advbench", "alpaca_benign"]
    df = build_results_dataframe(scores, labels, prompts, generations, dataset_names)
    assert set(df.columns) >= {
        "prompt", "generation", "label", "dataset",
        "is_refusal", "partial_compliance", "regex_hit", "judge_hit", "judge_rationale",
    }
    assert len(df) == 2


def test_compute_summary_stats():
    # 6 harmful, 4 benign — 4 harmful refusals, 1 benign refusal
    harmful_scores = [_make_score(True)] * 4 + [_make_score(False)] * 2
    benign_scores = [_make_score(True)] + [_make_score(False)] * 3
    all_scores = harmful_scores + benign_scores
    labels = ["harmful"] * 6 + ["benign"] * 4
    prompts = ["p"] * 10
    generations = ["g"] * 10
    datasets = ["advbench"] * 6 + ["alpaca_benign"] * 4
    df = build_results_dataframe(all_scores, labels, prompts, generations, datasets)
    stats = compute_summary_stats(df)
    assert stats["refusal_rate_harmful"] == pytest.approx(4 / 6)
    assert stats["false_refusal_rate_benign"] == pytest.approx(1 / 4)
    assert stats["n_total"] == 10
