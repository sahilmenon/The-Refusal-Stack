"""Tests for WandbLogger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("wandb", reason="wandb not installed in this environment")

from refusal_stack.eval.config import EvalConfig


@pytest.fixture()
def config():
    return EvalConfig(
        model_id="test/model",
        datasets=["advbench"],
        held_out_seed=42,
    )


def test_log_summary_prefixes_keys(config):
    with patch("wandb.init"), patch("wandb.log") as mock_log, patch("wandb.finish"):
        from refusal_stack.eval.wandb_logger import WandbLogger

        wl = WandbLogger(config, run_name="test")
        wl.log_summary({"refusal_rate_harmful": 0.9, "asr": 0.1})
        call_args = mock_log.call_args[0][0]
        assert "eval/refusal_rate_harmful" in call_args
        assert "eval/asr" in call_args


def test_log_results_artifact_creates_dataset_artifact(config, tmp_path):
    df = pd.DataFrame({"prompt": ["p"], "label": ["harmful"], "is_refusal": [True]})

    artifact_mock = MagicMock()
    artifact_mock.__enter__ = MagicMock(return_value=MagicMock())
    artifact_mock.__exit__ = MagicMock(return_value=False)
    artifact_mock.new_file = MagicMock(return_value=artifact_mock)

    with (
        patch("wandb.init"),
        patch("wandb.log"),
        patch("wandb.finish"),
        patch("wandb.Artifact", return_value=artifact_mock) as mock_art,
        patch("wandb.log_artifact"),
    ):
        from refusal_stack.eval.wandb_logger import WandbLogger

        wl = WandbLogger(config)
        wl.log_results_artifact(df)
        mock_art.assert_called_once_with("eval_results", type="dataset")


def test_finish_called(config):
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish") as mock_fin:
        from refusal_stack.eval.wandb_logger import WandbLogger

        wl = WandbLogger(config)
        wl.finish()
        mock_fin.assert_called_once()
