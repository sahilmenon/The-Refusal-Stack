"""Result types for Phase 5 agentic eval."""

from __future__ import annotations

from pydantic import BaseModel


class AgenticEvalResult(BaseModel):
    refusal_rate: float
    asr: float
    per_category_refusal: dict[str, float] = {}
    mean_turns_to_refusal: float = 0.0
    tool_call_counts: dict[str, int] = {}
    run_id: str = ""


class AgenticAttackResult(BaseModel):
    attack_type: str
    goal: str
    n_turns: int
    asr: float
    successful_turns: list[int] = []
    conversation: dict = {}


class DeltaReport(BaseModel):
    refusal_rate_delta: float
    asr_delta: float
    per_category_delta: dict[str, float] = {}
    attack_delta: dict[str, float] = {}
    mean_turns_to_refusal: float = 0.0
