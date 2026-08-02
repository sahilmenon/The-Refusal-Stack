"""Tests for Phase 4 SFT data construction."""
from __future__ import annotations


def _make_harmful_examples(n: int = 20) -> list[dict]:
    return [{"prompt": f"how to do bad thing {i}", "completion": f"Sure, here is how to do bad thing {i}."} for i in range(n)]


def _make_benign_examples(n: int = 20) -> list[dict]:
    return [{"prompt": f"Explain topic {i}", "completion": f"Topic {i} involves..."} for i in range(n)]


def test_no_leak():
    from refusal_stack.finetune.data import train_test_split_no_leak
    examples = _make_harmful_examples(30)
    train, held_out = train_test_split_no_leak(examples, 15, 10, seed=42)
    train_prompts = {e["prompt"] for e in train}
    held_out_prompts = {e["prompt"] for e in held_out}
    assert len(train_prompts & held_out_prompts) == 0


def test_split_sizes():
    from refusal_stack.finetune.data import train_test_split_no_leak
    examples = _make_harmful_examples(30)
    train, held_out = train_test_split_no_leak(examples, 15, 10, seed=42)
    assert len(train) == 15
    assert len(held_out) == 10


def test_benign_no_harmful_keywords():
    examples = _make_benign_examples(20)
    harmful_keywords = ["bomb", "malware", "poison"]
    for ex in examples:
        text = (ex["prompt"] + ex["completion"]).lower()
        for kw in harmful_keywords:
            assert kw not in text, f"Found '{kw}' in benign example"


def test_build_hf_dataset():
    from unittest.mock import MagicMock

    from refusal_stack.finetune.data import build_hf_dataset
    examples = _make_harmful_examples(5)
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<s>formatted text</s>")
    ds = build_hf_dataset(examples, tokenizer)
    assert "text" in ds.column_names
    assert "prompt" in ds.column_names
    assert len(ds) == 5
