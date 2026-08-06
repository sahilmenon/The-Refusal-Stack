"""Tests for HFModelWrapper - all model and tokenizer calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this environment")
pytest.importorskip("transformers", reason="transformers not installed in this environment")


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.eos_token_id = 2
    tok.apply_chat_template = MagicMock(return_value="<|user|>test prompt<|assistant|>")
    # Encode returns a batch of tokens
    tok.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    tok.decode = MagicMock(return_value="test generation")
    return tok


def _make_mock_model():
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    params_iter = iter([torch.zeros(1)])
    model.parameters = MagicMock(return_value=params_iter)
    # generate returns shape (1, 10) - 5 input + 5 new tokens
    model.generate = MagicMock(return_value=torch.zeros(1, 10, dtype=torch.long))
    return model


@pytest.fixture()
def wrapper(tmp_path):
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_make_mock_tokenizer()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_make_mock_model()),
    ):
        from refusal_stack.eval.model_wrapper import HFModelWrapper

        w = HFModelWrapper("test/model", max_new_tokens=10, temperature=0.0)
        return w


def test_chat_template_applied(wrapper):
    """apply_chat_template is called during generate_batch."""
    wrapper.tokenizer.apply_chat_template.reset_mock()
    wrapper.tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    wrapper.tokenizer.side_effect = None
    wrapper.generate_batch(["test prompt"])
    wrapper.tokenizer.apply_chat_template.assert_called_once()


def test_empty_generation_replaced(wrapper):
    """Empty string after decode → [EMPTY_GENERATION]."""
    wrapper.tokenizer.decode = MagicMock(return_value="")
    wrapper.tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    result = wrapper.generate_batch(["a prompt"])
    assert result[0] == "[EMPTY_GENERATION]"


def test_cache_hit_skips_model_generate(wrapper, tmp_path):
    """A cached generation is returned without calling model.generate."""
    from refusal_stack.eval.cache import GenerationCache

    cache = GenerationCache(str(tmp_path), "test/model", "advbench")
    key = cache.make_key("test/model", None, "cached prompt")
    cache.set(key, "cached generation")
    wrapper._cache = cache
    wrapper.model.generate.reset_mock()
    result = wrapper.generate_batch(["cached prompt"])
    assert result == ["cached generation"]
    wrapper.model.generate.assert_not_called()
