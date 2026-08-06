"""Unit tests for PAIR LLM-client factory - CPU-safe paths only.

The local provider loads a real HF model, so we never exercise that path here
(transformers is not installed on the CPU test box). We only assert the
unknown-provider guard and the factory's basic contract.
"""

from __future__ import annotations

import pytest

from refusal_stack.attacks.pair_clients import LLMClient, make_client


def test_make_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        make_client("fake/model", provider="does-not-exist")


def test_make_client_unknown_provider_names_the_provider():
    with pytest.raises(ValueError) as excinfo:
        make_client("fake/model", provider="openai-typo")
    assert "openai-typo" in str(excinfo.value)


def test_llmclient_protocol_is_runtime_checkable():
    # A duck-typed object with a matching chat() satisfies the Protocol.
    class _Dummy:
        def chat(self, messages, temperature, max_tokens):
            return "ok"

    assert isinstance(_Dummy(), LLMClient)

    class _NotAClient:
        pass

    assert not isinstance(_NotAClient(), LLMClient)
