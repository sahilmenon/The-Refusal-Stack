"""Inspect AI solver for multi-turn agentic eval."""
from __future__ import annotations

from refusal_stack.agent.config import AgentConfig


def build_inspect_solver(config: AgentConfig):
    try:
        from inspect_ai.solver import basic_agent, system_message
        from inspect_ai.tool import tool

        return basic_agent(
            max_messages=config.max_turns * 2,
            system_message=config.system_prompt,
        )
    except ImportError:
        return None
