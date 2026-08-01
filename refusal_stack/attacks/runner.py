from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from refusal_stack.attacks.config import GCGConfig, PAIRConfig
from refusal_stack.attacks.utils import set_deterministic_mode
from refusal_stack.eval.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="refusal-attacks")
    parser.add_argument("--attack", choices=["gcg", "pair", "both"], default="both")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts-override", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    try:
        gcg_config = GCGConfig.from_yaml(args.config)
    except Exception:
        gcg_config = GCGConfig()

    if args.prompts_override:
        gcg_config.max_prompts = args.prompts_override
    if args.dry_run:
        gcg_config.n_steps = 1
        gcg_config.max_prompts = 2

    set_deterministic_mode(gcg_config.seed)

    from refusal_stack.attacks.gcg_data import load_gcg_dataset
    dataset = load_gcg_dataset(gcg_config)

    Path("figures").mkdir(exist_ok=True)
    from refusal_stack.attacks.figures import (
        make_asr_bar_chart, make_queries_to_success_cdf,
        make_attack_success_over_iterations, make_headroom_figure
    )

    gcg_results, pair_results = [], []

    if args.attack in ("gcg", "both") and not args.dry_run:
        logger.info("Running GCG on %d prompts", len(dataset))
        from refusal_stack.attacks.gcg import GCGAttack
        attacker = GCGAttack(gcg_config)
        for item in dataset:
            r = attacker.run(item["prompt"], item.get("target", "Sure"))
            gcg_results.append(r)
            logger.info("GCG result: success=%s", r.success)
    elif args.dry_run:
        from refusal_stack.attacks.base import AttackResult
        gcg_results = [AttackResult("test", "suffix", "target", False, 1.0, 1, 1, "gcg", "test")]

    if args.attack in ("pair", "both") and not args.dry_run:
        pair_config = PAIRConfig(max_prompts=gcg_config.max_prompts, seed=gcg_config.seed)
        from refusal_stack.attacks.pair import PAIRAttack
        attacker = PAIRAttack(pair_config)
        for item in dataset:
            r = attacker.run(item["prompt"], item.get("target", ""))
            pair_results.append(r)
    elif args.dry_run:
        from refusal_stack.attacks.base import AttackResult
        pair_results = [AttackResult("test", "prompt", "target", False, 1.0, 2, 2, "pair", "test")]

    from refusal_stack.attacks.analysis import run_analysis
    analysis = run_analysis(gcg_results, pair_results, gcg_config)
    logger.info("Analysis: %s", json.dumps({k: v for k, v in analysis.items() if isinstance(v, (int, float))}, indent=2))

    make_asr_bar_chart(
        baseline_asr=0.05, gcg_asr=analysis.get("gcg_asr", 0.0),
        pair_asr=analysis.get("pair_asr", 0.0), out_path=Path("figures/asr_bar_chart.png")
    )
    make_queries_to_success_cdf(gcg_results, pair_results, Path("figures/queries_to_success_cdf.png"))
    make_attack_success_over_iterations([], Path("figures/attack_success_over_iterations.png"))
    make_headroom_figure(analysis, Path("figures/headroom_figure.png"))

    results_path = Path("results/phase2_attacks.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"gcg": [vars(r) for r in gcg_results], "pair": [vars(r) for r in pair_results], **analysis}, f, indent=2, default=str)

    logger.info("Done. Results written to %s", results_path)


if __name__ == "__main__":
    main()
