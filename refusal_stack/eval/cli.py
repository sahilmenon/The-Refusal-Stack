"""CLI entrypoint for the refusal evaluation harness.

Usage:
    python -m refusal_stack.eval.cli --config configs/eval_base.yaml [options]

Or via the installed script:
    refusal-eval --config configs/eval_base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refusal-eval",
        description="Evaluate refusal behavior on an open-weight LLM.",
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--model-id", help="Override config.model_id")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM judge, regex only")
    p.add_argument("--datasets", nargs="+", help="Override config.datasets")
    p.add_argument("--output-dir", help="Override config.output_dir")
    p.add_argument("--dry-run", action="store_true", help="Generate 5 samples, skip W&B + figures")
    p.add_argument("--cache-only", action="store_true", help="Recompute metrics from cache only")
    p.add_argument("--seed", type=int, help="Override config.held_out_seed")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default=None)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    overrides: dict = {}
    if args.model_id:
        overrides["model_id"] = args.model_id
    if args.datasets:
        overrides["datasets"] = args.datasets
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["held_out_seed"] = args.seed
    if args.log_level:
        overrides["log_level"] = args.log_level
    raw.update(overrides)

    from refusal_stack.eval.config import load_eval_config
    from refusal_stack.eval.utils import configure_logging, set_all_seeds

    config = load_eval_config(args.config, overrides)
    configure_logging(config.log_level)
    logger = logging.getLogger(__name__)
    set_all_seeds(config.held_out_seed)

    logger.info("Starting eval run: model=%s, datasets=%s", config.model_id, config.datasets)

    try:
        _run_eval(args, config, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
        sys.exit(1)
    except Exception:
        logger.exception("Eval failed with unhandled exception")
        raise


def _run_eval(args, config, logger) -> None:

    from refusal_stack.data.loaders import load_eval_datasets
    from refusal_stack.eval.cache import GenerationCache
    from refusal_stack.eval.figures import save_all_figures
    from refusal_stack.eval.metrics import (
        build_results_dataframe,
        compute_summary_stats,
        per_dataset_breakdown,
    )
    from refusal_stack.eval.scorers import score_batch
    from refusal_stack.eval.wandb_logger import WandbLogger

    # Data
    logger.info("Loading datasets...")
    datasets = load_eval_datasets(config)

    # Model
    if not args.cache_only:
        logger.info("Loading model: %s", config.model_id)
        from refusal_stack.eval.cache import GenerationCache
        from refusal_stack.eval.model_wrapper import HFModelWrapper

        wrapper = HFModelWrapper(
            model_id=config.model_id,
            revision=getattr(config, "model_revision", None),
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        wrapper = None

    # Judge
    judge_fn = None
    if not args.no_judge and config.judge_provider == "local":
        try:
            from refusal_stack.eval.judge import LocalJudge

            judge_fn = LocalJudge(config.judge_model)
            logger.info("Using local judge: %s", config.judge_model)
        except Exception as exc:
            logger.warning("LocalJudge unavailable (%s) - regex-only scoring", exc)

    # Per-dataset generation + scoring
    all_scores = []
    all_labels: list[str] = []
    all_prompts: list[str] = []
    all_generations: list[str] = []
    all_dataset_names: list[str] = []

    for ds_name, ds in datasets.items():
        logger.info("Processing dataset: %s (%d rows)", ds_name, len(ds))
        prompts = ds["prompt"]
        labels = ds["label"]

        if args.dry_run:
            prompts = prompts[:5]
            labels = labels[:5]

        if args.cache_only or wrapper is None:
            # Try to load from cache; use placeholder if not found
            cache = GenerationCache(config.cache_dir, config.model_id, ds_name)
            generations = []
            for p in prompts:
                key = cache.make_key(config.model_id, None, p)
                gen = cache.get(key) or "[CACHE_MISS]"
                generations.append(gen)
        else:
            cache = GenerationCache(config.cache_dir, config.model_id, ds_name)
            wrapper._cache = cache
            generations = []
            for i in range(0, len(prompts), config.batch_size):
                batch_p = prompts[i : i + config.batch_size]
                batch_g = wrapper.generate_batch(batch_p, seed=config.held_out_seed)
                generations.extend(batch_g)
                if args.dry_run:
                    break

        scores = score_batch(
            list(prompts),
            generations,
            run_judge=judge_fn is not None,
            judge_fn=judge_fn,
        )
        all_scores.extend(scores)
        all_labels.extend(labels)
        all_prompts.extend(prompts)
        all_generations.extend(generations)
        all_dataset_names.extend([ds_name] * len(scores))

    # Metrics
    df = build_results_dataframe(
        all_scores, all_labels, all_prompts, all_generations, all_dataset_names
    )
    stats = compute_summary_stats(df)
    breakdown = per_dataset_breakdown(df)

    # Print summary
    _print_summary(stats, breakdown)

    if args.dry_run:
        logger.info("Dry run complete; skipping W&B + figures")
        return

    # Save results JSON
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = config.model_id.replace("/", "_")
    results_path = results_dir / f"phase1_eval_{slug}_{ts}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Also write the canonical, un-timestamped path that downstream phases
    # (delta analysis, repro checks, figure scripts) load by fixed name.
    canonical_path = results_dir / "phase1_eval.json"
    with canonical_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("Canonical results written to %s", canonical_path)

    # W&B
    run_name = f"eval_{slug}_{ts}"
    wandb_logger = WandbLogger(config, run_name=run_name)
    wandb_logger.log_summary(stats)
    wandb_logger.log_per_dataset(breakdown)
    wandb_logger.log_results_artifact(df)
    wandb_logger.finish()

    # Figures
    logger.info("Saving figures to %s", config.output_dir)
    save_all_figures(df, config.output_dir)


def _print_summary(stats: dict, breakdown) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Eval Summary")
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in stats.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) and v == v else str(v))
        console.print(table)
    except ImportError:
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\nPer-dataset breakdown:")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
