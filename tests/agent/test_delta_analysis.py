"""Tests for delta analysis."""
from __future__ import annotations

import numpy as np
import pytest

from refusal_stack.agent.delta_analysis import bootstrap_ci, compute_delta
from refusal_stack.agent.results import AgenticEvalResult


def test_refusal_rate_delta():
    single = {"refusal_rate_harmful": 0.9, "asr": 0.1}
    agentic = AgenticEvalResult(refusal_rate=0.7, asr=0.3)
    report = compute_delta(single, agentic)
    assert abs(report.refusal_rate_delta - (-0.2)) < 1e-4


def test_ci_bounds_bracket_zero():
    rng = np.random.default_rng(42)
    deltas = list(rng.normal(0, 1, 100))
    lo, hi = bootstrap_ci(deltas, n_bootstrap=500, seed=0)
    assert lo < 0 < hi


def test_ci_positive_distribution():
    deltas = [1.0] * 50
    lo, hi = bootstrap_ci(deltas, n_bootstrap=200, seed=0)
    assert lo > 0


def test_delta_report_serializable():
    import json
    from refusal_stack.agent.delta_report import save_delta_json
    from refusal_stack.agent.results import DeltaReport
    import tempfile, os

    report = DeltaReport(refusal_rate_delta=-0.1, asr_delta=0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results", "delta.json")
        save_delta_json(report, path)
        with open(path) as f:
            data = json.load(f)
    assert "refusal_rate_delta" in data
