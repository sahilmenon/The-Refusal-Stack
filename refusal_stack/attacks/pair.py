from __future__ import annotations

import logging
from dataclasses import dataclass, field

from refusal_stack.attacks.base import AttackResult, BaseAttack
from refusal_stack.attacks.config import PAIRConfig
from refusal_stack.attacks.pair_clients import make_client
from refusal_stack.attacks.pair_prompts import (
    ATTACKER_SYSTEM_PROMPT,
    FOLLOWUP_ATTACKER_USER_TEMPLATE,
    INITIAL_ATTACKER_USER_TEMPLATE,
    PAIRParseError,
    parse_attacker_response,
)

logger = logging.getLogger(__name__)


@dataclass
class PAIRStream:
    history: list[dict] = field(default_factory=list)
    best_score: float = 0.0
    best_adversarial_prompt: str = ""
    best_response: str = ""  # the TARGET's response to the best prompt (what we score)
    queries_used: int = 0


class PAIRAttack(BaseAttack):
    def __init__(self, config: PAIRConfig):
        super().__init__(config)
        self.config = config
        self._attacker = make_client(config.attacker_model, provider="local")
        self._target = make_client(config.target_model, provider="local")
        # Llama-Guard is a safe/unsafe classifier, not a 1-10 rater - drive it
        # through LocalJudge (which applies Guard's own template and reads the
        # safe/unsafe verdict) and use it as the paper's binary judge.
        from refusal_stack.eval.judge import LocalJudge

        self._judge = LocalJudge(config.judge_model)

    def run(self, prompt: str, target: str) -> AttackResult:
        from refusal_stack.eval import score_generation

        streams = [PAIRStream() for _ in range(self.config.n_streams)]
        query_budget = self.config.max_queries // self.config.n_streams

        for stream_id, stream in enumerate(streams):
            stream = self._run_stream(stream, prompt, target, query_budget, stream_id)
            streams[stream_id] = stream
            if stream.best_score >= self.config.judge_threshold:
                break

        success = any(s.best_score >= self.config.judge_threshold for s in streams)
        total_queries = sum(s.queries_used for s in streams)
        best = max(streams, key=lambda s: s.best_score)

        # Score the TARGET's response to the best jailbreak - NOT the attacker's
        # prompt (which contains no refusal markers and would always read as compliant).
        rs = score_generation(prompt, best.best_response or "")
        return AttackResult(
            prompt=prompt,
            adversarial_string=best.best_adversarial_prompt,
            target=target,
            success=success,
            score=best.best_score,
            queries=total_queries,
            iterations=total_queries,
            attack_type="pair",
            model_id=self.config.target_model,
            metadata={"pair_judge_score": best.best_score, "harness_score": rs.is_refusal},
        )

    def _run_stream(
        self, stream: PAIRStream, goal: str, target_str: str, budget: int, stream_id: int
    ) -> PAIRStream:
        stagnant = 0
        prev_prompt = ""
        prev_response = ""
        prev_score = 0.0

        for q in range(budget):
            if q == 0:
                user_msg = INITIAL_ATTACKER_USER_TEMPLATE.format(goal=goal, target_str=target_str)
            else:
                user_msg = FOLLOWUP_ATTACKER_USER_TEMPLATE.format(
                    goal=goal,
                    target_str=target_str,
                    prev_prompt=prev_prompt,
                    prev_response=prev_response[:200],
                    score=int(prev_score),
                )

            attacker_msgs = [
                {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
                *stream.history,
                {"role": "user", "content": user_msg},
            ]

            attacker_raw = ""
            try:
                attacker_raw = self._attacker.chat(
                    attacker_msgs, self.config.temperature_attacker, self.config.max_tokens_attacker
                )
                parsed = parse_attacker_response(attacker_raw)
                adv_prompt = parsed.get("prompt", goal)
            except (PAIRParseError, Exception) as e:
                logger.warning("Attacker parse failed (stream %d, q %d): %s", stream_id, q, e)
                adv_prompt = goal

            target_response = self._target.chat(
                [{"role": "user", "content": adv_prompt}],
                self.config.temperature_target,
                self.config.max_tokens_target,
            )

            # Binary judge (Chao et al. Algorithm 1, S in {0,1}): Llama-Guard says
            # "unsafe" -> the target complied with the harmful goal -> jailbroken.
            # judged_refusal is False (unsafe), True (safe/refused), or None (parse
            # failure); only a definite "unsafe" counts as success.
            try:
                judged_refusal, _ = self._judge(goal, target_response)
                score = 10.0 if judged_refusal is False else 1.0
            except Exception as e:
                logger.warning("Judge failed: %s", e)
                score = 1.0

            stream.queries_used += 1
            # Append BOTH turns so the attacker's context is a valid alternating
            # transcript of its prompts + the judge feedback it must refine from.
            stream.history.append({"role": "user", "content": user_msg})
            stream.history.append({"role": "assistant", "content": attacker_raw})

            if score > stream.best_score:
                stream.best_score = score
                stream.best_adversarial_prompt = adv_prompt
                stream.best_response = target_response
                stagnant = 0
            else:
                stagnant += 1

            prev_prompt = adv_prompt
            prev_response = target_response
            prev_score = score

            if score >= self.config.judge_threshold:
                break

            if stagnant >= self.config.stagnation_patience:
                stream.history = []
                stagnant = 0

        return stream

    def run_batch(self, prompts: list[str], targets: list[str]) -> list[AttackResult]:
        return [self.run(p, t) for p, t in zip(prompts, targets)]
