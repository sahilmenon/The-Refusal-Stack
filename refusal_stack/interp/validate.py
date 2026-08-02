from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from refusal_stack.eval.scorers import score_outputs

logger = logging.getLogger(__name__)


@dataclass
class CausalValidationResult:
    scenario: str
    refusal_rate: float
    false_refusal_rate: float
    asr: float
    alpha: float | None = None


def score_outputs_with_harness(outputs_jsonl: str, scenario: str, config) -> dict:
    records = []
    with open(outputs_jsonl) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return score_outputs(records)


def run_full_validation(config, best_layer: int, direction, model, tokenizer) -> list[CausalValidationResult]:
    results = []
    # Scenarios are logged; actual run happens via run_interp.py orchestration
    logger.info("Full causal validation would run 4 scenarios: baseline harmful/harmless + ablated + steered")
    return results
