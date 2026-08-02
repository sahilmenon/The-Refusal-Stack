"""Tests for the LLM judge module."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from refusal_stack.eval.judge import llm_judge_score


def _make_mock_anthropic_client(response_text: str):
    client = AsyncMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create = AsyncMock(return_value=msg)
    return client


def test_judge_returns_true_for_refusal():
    client = _make_mock_anthropic_client(
        json.dumps({"is_refusal": True, "rationale": "model declined"})
    )
    result = asyncio.run(
        llm_judge_score("harmful prompt", "I can't help", "claude-3", "anthropic", client)
    )
    assert result == (True, "model declined")


def test_judge_returns_false_for_compliance():
    client = _make_mock_anthropic_client(
        json.dumps({"is_refusal": False, "rationale": "model complied"})
    )
    result = asyncio.run(
        llm_judge_score("harmful prompt", "Sure, here you go", "claude-3", "anthropic", client)
    )
    assert result == (False, "model complied")


def test_judge_retries_on_malformed_json():
    """First call returns garbage JSON; second call succeeds."""
    call_count = 0
    valid_response = json.dumps({"is_refusal": True, "rationale": "retry worked"})

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        msg = MagicMock()
        if call_count == 1:
            msg.content = [MagicMock(text="NOT JSON AT ALL")]
        else:
            msg.content = [MagicMock(text=valid_response)]
        return msg

    client = AsyncMock()
    client.messages.create = side_effect
    result = asyncio.run(
        llm_judge_score("prompt", "response", "claude-3", "anthropic", client)
    )
    assert result == (True, "retry worked")
    assert call_count == 2


def test_judge_parse_error_after_two_failures():
    """Two consecutive JSON failures → JUDGE_PARSE_ERROR."""
    client = _make_mock_anthropic_client("GARBAGE")
    result = asyncio.run(
        llm_judge_score("prompt", "response", "claude-3", "anthropic", client)
    )
    assert result == (False, "JUDGE_PARSE_ERROR")
