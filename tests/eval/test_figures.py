"""Tests for figure generation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("plotly", reason="plotly not installed in this environment")

from refusal_stack.eval.metrics import build_results_dataframe
from refusal_stack.eval.scorers import RefusalScore


def _sample_df():
    scores = [
        RefusalScore(True, False, True, raw_text="I can't help"),
        RefusalScore(False, False, False, raw_text="Sure!"),
        RefusalScore(True, True, True, raw_text="Sorry but here is how: step 1"),
        RefusalScore(False, False, False, raw_text="Here you go"),
    ]
    labels = ["harmful", "harmful", "harmful", "benign"]
    prompts = ["p1", "p2", "p3", "p4"]
    gens = ["I can't help", "Sure!", "Sorry step 1", "Here you go"]
    datasets = ["advbench", "advbench", "advbench", "alpaca_benign"]
    return build_results_dataframe(scores, labels, prompts, gens, datasets)


def test_plot_refusal_rates_returns_figure(tmp_path):
    import plotly.graph_objects as go

    from refusal_stack.eval.figures import plot_refusal_rates

    df = _sample_df()
    fig, path = plot_refusal_rates(df, str(tmp_path))
    assert isinstance(fig, go.Figure)
    assert Path(path).exists()


def test_plot_score_distribution_returns_figure(tmp_path):
    import plotly.graph_objects as go

    from refusal_stack.eval.figures import plot_score_distribution

    df = _sample_df()
    fig, path = plot_score_distribution(df, str(tmp_path))
    assert isinstance(fig, go.Figure)


def test_save_all_figures_returns_paths(tmp_path):
    from refusal_stack.eval.figures import save_all_figures

    df = _sample_df()
    paths = save_all_figures(df, str(tmp_path))
    assert len(paths) >= 1  # at least one succeeded
