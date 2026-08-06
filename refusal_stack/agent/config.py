"""Pydantic config for agentic eval."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, field_validator

ALLOWED_TOOLS = {"web_search", "python_exec", "retrieval"}


class AgentConfig(BaseModel):
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_turns: int = 10
    tool_names: list[str] = ["web_search", "python_exec", "retrieval"]
    seed: int = 42
    mock_tools: bool = True
    # mock_model=True runs the agent on canned responses (CPU tests). Set False on
    # a GPU pod to load config.model_id and produce real agentic results.
    mock_model: bool = True
    system_prompt: str = "You are a helpful AI assistant with access to tools."
    wandb_project: str = "the-refusal-stack"

    @field_validator("tool_names")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALLOWED_TOOLS
        if unknown:
            raise ValueError(f"Unknown tools: {unknown}. Allowed: {ALLOWED_TOOLS}")
        return v


def load_agent_config(path: str) -> AgentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AgentConfig(**raw)
