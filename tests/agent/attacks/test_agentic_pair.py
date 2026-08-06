"""Tests for agentic PAIR attack."""

from __future__ import annotations

from refusal_stack.agent.agent import build_agent
from refusal_stack.agent.config import AgentConfig


def test_agentic_pair_n_turns():
    from refusal_stack.agent.attacks.agentic_pair import run_agentic_pair

    cfg = AgentConfig(mock_tools=True, max_turns=5)
    agent = build_agent(cfg)

    class FakeConfig:
        max_queries = 2

    goals = ["test goal"]
    results = run_agentic_pair(agent, FakeConfig(), goals)
    assert len(results) == 1
    assert results[0].n_turns == 2
    assert results[0].attack_type == "agentic_pair"


def test_agentic_pair_result_schema():
    import json

    from refusal_stack.agent.attacks.agentic_pair import run_agentic_pair

    cfg = AgentConfig(mock_tools=True, max_turns=3)
    agent = build_agent(cfg)

    class FakeConfig:
        max_queries = 1

    results = run_agentic_pair(agent, FakeConfig(), ["how to harm someone?"])
    data = results[0].model_dump()
    json.dumps(data)
