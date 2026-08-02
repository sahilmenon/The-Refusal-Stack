from __future__ import annotations

import argparse
import json
import logging
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
    parser.add_argument(
        "--out", default="results/phase2_attacks.json",
        help="Results path; override so a validation run (e.g. Vicuna) does not "
             "clobber the primary Llama-3.1 baseline.",
    )
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

    # Best-effort W&B run — never let tracking break the attack itself.
    wandb_run = None
    if not args.dry_run:
        try:
            from refusal_stack.attacks.wandb_utils import init_attack_run
            wandb_run = init_attack_run(gcg_config, args.attack)
            logger.info("W&B run initialized: %s", getattr(wandb_run, "name", "?"))
        except Exception as exc:  # noqa: BLE001 — offline / no-wandb is fine
            logger.warning("W&B unavailable (%s) — continuing without tracking", exc)

    from refusal_stack.attacks.gcg_data import load_gcg_dataset
    dataset = load_gcg_dataset(gcg_config)

    Path("figures").mkdir(exist_ok=True)
    from refusal_stack.attacks.figures import (
        make_asr_bar_chart,
        make_attack_success_over_iterations,
        make_headroom_figure,
        make_queries_to_success_cdf,
    )

    gcg_results, pair_results = [], []

    if args.attack in ("gcg", "both") and not args.dry_run:
        logger.info("Running GCG on %d prompts", len(dataset))
        from refusal_stack.attacks.gcg import GCGAttack
        attacker = GCGAttack(gcg_config)
        n = len(dataset)
        n_success = 0
        for i, item in enumerate(dataset, 1):
            logger.info("GCG prompt %d/%d: %s", i, n, item["prompt"][:60])
            r = attacker.run(item["prompt"], item.get("target", "Sure"))
            gcg_results.append(r)
            n_success += int(r.success)
            logger.info("GCG prompt %d/%d done: success=%s | running ASR=%.3f (%d/%d)",
                        i, n, r.success, n_success / i, n_success, i)
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

    figure_paths = [
        Path("figures/asr_bar_chart.png"),
        Path("figures/queries_to_success_cdf.png"),
        Path("figures/attack_success_over_iterations.png"),
        Path("figures/headroom_figure.png"),
    ]
    make_asr_bar_chart(
        baseline_asr=0.05, gcg_asr=analysis.get("gcg_asr", 0.0),
        pair_asr=analysis.get("pair_asr", 0.0), out_path=figure_paths[0]
    )
    make_queries_to_success_cdf(gcg_results, pair_results, figure_paths[1])
    make_attack_success_over_iterations([], figure_paths[2])
    make_headroom_figure(analysis, figure_paths[3])

    results_path = Path(args.out)
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"gcg": [vars(r) for r in gcg_results], "pair": [vars(r) for r in pair_results], **analysis}, f, indent=2, default=str)

    logger.info("Done. Results written to %s", results_path)

    # Log headline metrics, successful-string tables, and figures to W&B.
    if wandb_run is not None:
        try:
            from refusal_stack.attacks.wandb_utils import log_figures, log_successful_strings
            wandb_run.log({f"attack/{k}": v for k, v in analysis.items() if isinstance(v, (int, float))})
            log_successful_strings(wandb_run, gcg_results, "gcg")
            log_successful_strings(wandb_run, pair_results, "pair")
            log_figures(wandb_run, figure_paths)
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("W&B logging failed (%s)", exc)


if __name__ == "__main__":
    main()
