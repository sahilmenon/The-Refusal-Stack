"""Tests for agent conversation state and solver."""
from __future__ import annotations

import pytest

from refusal_stack.agent.config import AgentConfig
from refusal_stack.agent.conversation_state import ConversationState


def test_conversation_state_add():
    state = ConversationState()
    state.add("user", "Hello")
    state.add("assistant", "Hi there")
    assert len(state) == 2
    assert state.messages[0].role == "user"


def test_conversation_state_to_dict():
    state = ConversationState()
    state.add("user", "test")
    d = state.to_dict()
    assert "messages" in d
    assert d["messages"][0]["role"] == "user"
    assert d["messages"][0]["content"] == "test"


def test_conversation_state_serializable():
    import json
    state = ConversationState()
    state.add("user", "hello")
    state.add("assistant", "world")
    json.dumps(state.to_dict())


def test_conversation_all_text():
    state = ConversationState()
    state.add("user", "user message")
    state.add("assistant", "assistant response")
    text = state.all_text()
    assert "user message" in text
    assert "assistant response" in text
