"""Tests for agentic eval runner."""
from __future__ import annotations

import json

from refusal_stack.agent.results import AgenticEvalResult


def test_agentic_eval_result_bounds():
    result = AgenticEvalResult(
        refusal_rate=0.8,
        asr=0.2,
        per_category_refusal={"harmful": 0.8},
        tool_call_counts={"web_search": 3},
    )
    assert 0.0 <= result.refusal_rate <= 1.0
    assert 0.0 <= result.asr <= 1.0


def test_agentic_eval_result_serializable():
    result = AgenticEvalResult(
        refusal_rate=0.5,
        asr=0.5,
        run_id="test-run-001",
    )
    data = result.model_dump()
    json.dumps(data)


def test_score_agentic_transcript_refusal():
    from refusal_stack.agent.conversation_state import ConversationState
    from refusal_stack.agent.eval_runner import score_agentic_transcript

    state = ConversationState()
    state.add("assistant", "I'm sorry, I cannot help with that.")
    result = score_agentic_transcript(state, scorer=None)
    assert result["is_refusal"] is True


def test_score_agentic_transcript_compliance():
    from refusal_stack.agent.conversation_state import ConversationState
    from refusal_stack.agent.eval_runner import score_agentic_transcript

    state = ConversationState()
    state.add("assistant", "Sure! Here is how you do it: step 1...")
    result = score_agentic_transcript(state, scorer=None)
    assert result["is_refusal"] is False
