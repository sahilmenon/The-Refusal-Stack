"""Compute single-turn vs agentic delta metrics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from refusal_stack.agent.results import AgenticEvalResult, DeltaReport

log = logging.getLogger(__name__)


def bootstrap_ci(
    delta_series: list[float], n_bootstrap: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(delta_series)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_delta(single_turn_result: dict, agentic_result: AgenticEvalResult) -> DeltaReport:
    rr_delta = agentic_result.refusal_rate - single_turn_result.get("refusal_rate_harmful", 0.0)
    asr_delta = agentic_result.asr - single_turn_result.get("asr", 0.0)
    per_cat = {
        k: agentic_result.per_category_refusal.get(k, 0.0) - single_turn_result.get(f"per_category_{k}", 0.0)
        for k in agentic_result.per_category_refusal
    }
    return DeltaReport(
        refusal_rate_delta=rr_delta,
        asr_delta=asr_delta,
        per_category_delta=per_cat,
        mean_turns_to_refusal=agentic_result.mean_turns_to_refusal,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", required=True)
    parser.add_argument("--agentic", required=True)
    parser.add_argument("--attacks-single", default=None)
    parser.add_argument("--attacks-agentic", default=None)
    parser.add_argument("--out", default="results/delta_report.json")
    args = parser.parse_args()

    with open(args.single) as f:
        single = json.load(f)
    with open(args.agentic) as f:
        agentic_raw = json.load(f)

    agentic = AgenticEvalResult(**agentic_raw)
    report = compute_delta(single, agentic)

    if args.attacks_single and args.attacks_agentic:
        with open(args.attacks_single) as f:
            atk_single = json.load(f)
        with open(args.attacks_agentic) as f:
            atk_agentic = json.load(f)
        for key in ["agentic_pair", "indirect_injection"]:
            if key in atk_agentic:
                agentic_asr = sum(r["asr"] for r in atk_agentic[key]) / max(len(atk_agentic[key]), 1)
                single_asr = atk_single.get(f"{key}_asr", 0.0)
                report.attack_delta[key] = agentic_asr - single_asr

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    log.info(f"Delta report saved to {out_path}")

    try:
        import wandb
        wandb.log({f"delta/{k}": v for k, v in report.model_dump().items() if isinstance(v, (int, float))})
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
