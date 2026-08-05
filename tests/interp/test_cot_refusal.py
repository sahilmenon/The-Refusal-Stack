"""CPU-only tests for 8B reasoning-model CoT refusal (arXiv:2507.03167).

No model / GPU: exercises the CoT/answer split, the token-boundary finder, the
'fires mid-CoT' decision, and config loading. The DeepSeek generation + capture
path in run_cot_refusal is pod-only and not tested here.
"""
import numpy as np
import pytest

from refusal_stack.interp.cot_refusal import (
    CoTConfig,
    cot_token_boundary,
    fires_before_answer,
    load_cot_config,
    split_cot_answer,
)


def test_split_with_full_think_block():
    resp = "<think>let me reason about this request carefully</think>I cannot help with that."
    s = split_cot_answer(resp)
    assert s.has_think is True
    assert "reason about this" in s.cot
    assert s.answer == "I cannot help with that."


def test_split_without_opening_tag():
    # Some R1 distills emit the CoT then </think> without an explicit opener.
    resp = "reasoning step one, reasoning step two</think>Final answer here."
    s = split_cot_answer(resp)
    assert s.has_think is True
    assert s.answer == "Final answer here."


def test_split_no_think_all_cot():
    resp = "the model never closed its reasoning trace"
    s = split_cot_answer(resp)
    assert s.has_think is False
    assert s.answer == ""
    assert s.cot == resp


def test_cot_token_boundary_found():
    full = [10, 11, 12, 90, 91, 20, 21]
    close = [90, 91]
    assert cot_token_boundary(full, close) == 3


def test_cot_token_boundary_absent_returns_len():
    full = [1, 2, 3]
    assert cot_token_boundary(full, [9, 9]) == 3


def test_fires_before_answer_positive_side():
    # harmful mean above harmless mean; a pre-boundary position crosses midpoint.
    proj = np.array([0.0, 0.1, 0.9, 0.2, 0.3])  # position 2 = 0.9 crosses
    assert fires_before_answer(proj, boundary_idx=3, harmless_mean=0.0, harmful_mean=1.0) is True


def test_fires_before_answer_no_crossing():
    proj = np.array([0.0, 0.1, 0.2])
    assert fires_before_answer(proj, boundary_idx=3, harmless_mean=0.0, harmful_mean=1.0) is False


def test_fires_before_answer_zero_boundary():
    proj = np.array([0.9, 0.9])
    assert fires_before_answer(proj, boundary_idx=0, harmless_mean=0.0, harmful_mean=1.0) is False


def test_load_cot_config_defaults(tmp_path):
    cfg = load_cot_config(str(tmp_path / "nope.yaml"))
    assert isinstance(cfg, CoTConfig)
    assert cfg.model_id == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def test_load_cot_config_reads_overrides(tmp_path):
    p = tmp_path / "cot.yaml"
    p.write_text("inherits: interp_base.yaml\nn_harmful: 10\nbest_layer: 12\n", encoding="utf-8")
    cfg = load_cot_config(str(p))
    assert cfg.n_harmful == 10
    assert cfg.best_layer == 12
