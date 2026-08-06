"""Agentic attack implementations."""

from refusal_stack.agent.attacks.agentic_pair import run_agentic_pair
from refusal_stack.agent.attacks.indirect_injection import run_indirect_injection

__all__ = ["run_agentic_pair", "run_indirect_injection"]
