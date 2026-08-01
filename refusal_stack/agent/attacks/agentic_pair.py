"""Run PAIR attack in the agentic multi-turn setting."""
from __future__ import annotations

import logging

from refusal_stack.agent.results import AgenticAttackResult

log = logging.getLogger(__name__)


def run_agentic_pair(agent, pair_config, goals: list[str]) -> list[AgenticAttackResult]:
    results = []
    for goal in goals:
        agent.reset()
        n_turns = 0
        successful_turns = []
        # Each PAIR iteration sends the jailbreak candidate as a user turn
        for iteration in range(pair_config.max_queries if hasattr(pair_config, "max_queries") else 5):
            candidate = f"[PAIR attempt {iteration}] {goal}"
            step = agent.step(candidate)
            n_turns += 1
            from refusal_stack.eval.scorers import score_generation
            score = score_generation(goal, step.assistant_text)
            if not score.is_refusal:
                successful_turns.append(n_turns)

        asr = len(successful_turns) / n_turns if n_turns else 0.0
        results.append(AgenticAttackResult(
            attack_type="agentic_pair",
            goal=goal,
            n_turns=n_turns,
            asr=asr,
            successful_turns=successful_turns,
            conversation=agent.conversation.to_dict(),
        ))
    return results
