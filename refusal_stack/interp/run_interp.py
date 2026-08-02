"""Phase 3 orchestrator: extract the refusal direction, probe, ablate, steer.

Runs the full interpretability pipeline end to end and produces the three
Phase-3 deliverables:
  1. artifacts/refusal_direction_latest.safetensors  (consumed by Phase 4)
  2. results/phase3_interp.json                       (canonical headline metrics)
  3. figures/*.png                                    (layer separation, probe, etc.)

The heavy stages require a GPU pod (a real Llama/Qwen forward pass); the staged
``--stage`` flag lets each step be run and resumed independently.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from refusal_stack.eval.utils import configure_logging
from refusal_stack.interp.config import load_interp_config
from refusal_stack.interp.utils import set_global_seed

logger = logging.getLogger(__name__)


def _refusal_rate(prompts: list[str], generations: list[str]) -> float:
    from refusal_stack.eval.scorers import score_generation

    if not generations:
        return float("nan")
    refusals = sum(score_generation(p, g).is_refusal for p, g in zip(prompts, generations))
    return refusals / len(generations)


def _baseline_generate(prompts: list[str], model, tokenizer, config) -> list[str]:
    import torch

    outs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)
        outs.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
    return outs


def run_pipeline(config, run_id: str, stage: str, force: bool) -> dict:
    from refusal_stack.interp.ablation import resolve_ablation_layers, run_ablated_generation
    from refusal_stack.interp.activation_cache import ActivationCacheReader, ActivationCacheWriter
    from refusal_stack.interp.dataset import (
        build_contrast_dataset,
        load_advbench_harmful,
        load_alpaca_benign,
        train_test_split_contrast,
    )
    from refusal_stack.interp.direction import (
        compute_layer_separation_score,
        extract_refusal_directions,
        select_best_layer,
    )
    from refusal_stack.interp.extract_activations import extract_activations_for_split
    from refusal_stack.interp.figures import (
        plot_ablation_refusal_rate,
        plot_cosine_sim_heatmap,
        plot_layer_separation,
        plot_probe_accuracy_per_layer,
        plot_steering_dose_response,
    )
    from refusal_stack.interp.model_loader import get_num_layers, load_model_and_tokenizer
    from refusal_stack.interp.probe import run_all_probes
    from refusal_stack.interp.steering import run_steered_generation

    for d in (config.artifact_dir, config.figures_dir, config.cache_dir, "results"):
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s", config.model_id)
    model, tokenizer = load_model_and_tokenizer(config.model_id)
    num_layers = get_num_layers(model)

    # --- Contrast dataset -----------------------------------------------------
    harmful = load_advbench_harmful(n=config.n_harmful, seed=config.seed)
    harmless = load_alpaca_benign(n=config.n_harmless, seed=config.seed)
    n = min(len(harmful), len(harmless))
    harmful_p, harmless_p = build_contrast_dataset(harmful[:n], harmless[:n], tokenizer, config)
    h_train, h_test, hl_train, hl_test = train_test_split_contrast(
        harmful_p, harmless_p, config.test_frac, config.seed
    )

    # --- Extract activations (train split) -----------------------------------
    writer = ActivationCacheWriter(config.cache_dir, run_id)
    reader = ActivationCacheReader(config.cache_dir, run_id)
    if stage in {"all", "extract"} and (force or not reader.exists(0, "harmful")):
        logger.info("Extracting activations for %d harmful + %d harmless prompts", len(h_train), len(hl_train))
        extract_activations_for_split(h_train, "harmful", model, tokenizer, config, writer, force)
        extract_activations_for_split(hl_train, "harmless", model, tokenizer, config, writer, force)
    if stage == "extract":
        return {"stage": "extract", "num_layers": num_layers}

    # --- Directions + layer selection ----------------------------------------
    directions = extract_refusal_directions(reader, num_layers, config)
    best_layer = select_best_layer(directions, reader)
    best_dir = directions[best_layer]
    sep_scores = {
        i: compute_layer_separation_score(
            reader.load_layer(i, "harmful"), reader.load_layer(i, "harmless"), d.vector
        )
        for i, d in directions.items()
    }
    plot_layer_separation(sep_scores, best_layer, str(Path(config.figures_dir) / "layer_separation_cohen_d.png"))

    # --- Linear probes --------------------------------------------------------
    probe_results = run_all_probes(directions, reader, config)
    plot_probe_accuracy_per_layer(probe_results, best_layer, str(Path(config.figures_dir) / "probe_accuracy_per_layer.png"))
    plot_cosine_sim_heatmap(probe_results, str(Path(config.figures_dir) / "probe_dom_cosine_heatmap.png"))

    # --- SAE feature alignment (§3J-SAE) -------------------------------------
    # Runs after probes, before persist. No-ops with a logged warning if sae-lens
    # can't load the Llama Scope release, so offline runs are unaffected.
    sae_summary = None
    if stage in {"all", "sae"}:
        sae_summary = _run_sae_stage(config, reader, best_dir, best_layer, h_test, model, tokenizer)
        if stage == "sae":
            return {"stage": "sae", "sae": sae_summary}

    # --- Persist artifact (Phase-4 hand-off) ---------------------------------
    artifact_path = save_artifact(best_dir, directions, config)

    # --- Causal validation: ablation + steering ------------------------------
    ablation_layers = resolve_ablation_layers(config.ablation_layer_strategy, best_layer, num_layers)
    baseline_gens = _baseline_generate(h_test, model, tokenizer, config)
    ablated_gens = run_ablated_generation(h_test, model, tokenizer, best_dir.vector, ablation_layers, config)
    baseline_rr = _refusal_rate(h_test, baseline_gens)
    ablated_rr = _refusal_rate(h_test, ablated_gens)
    plot_ablation_refusal_rate(baseline_rr, ablated_rr, str(Path(config.figures_dir) / "ablation_refusal_rate.png"))

    baseline_frr = _refusal_rate(hl_test, _baseline_generate(hl_test, model, tokenizer, config))
    steer_frrs = []
    for alpha in config.steering_alphas:
        steered = run_steered_generation(hl_test, model, tokenizer, best_dir.vector, [best_layer], alpha, config)
        steer_frrs.append(_refusal_rate(hl_test, steered))
    plot_steering_dose_response(
        list(config.steering_alphas), steer_frrs, baseline_frr,
        str(Path(config.figures_dir) / "steering_dose_response.png"),
    )

    # --- Canonical results JSON ----------------------------------------------
    summary = {
        "model_id": config.model_id,
        "num_layers": num_layers,
        "best_layer": best_layer,
        "best_layer_cohens_d": sep_scores[best_layer],
        "best_probe_accuracy": probe_results[best_layer].accuracy if best_layer in probe_results else None,
        "best_probe_dom_cosine": probe_results[best_layer].cosine_sim_vs_dom if best_layer in probe_results else None,
        "ablation_baseline_refusal_rate": baseline_rr,
        "ablation_refusal_rate": ablated_rr,
        "ablation_refusal_drop": baseline_rr - ablated_rr,
        "steering_baseline_false_refusal_rate": baseline_frr,
        "steering_alphas": list(config.steering_alphas),
        "steering_false_refusal_rates": steer_frrs,
        "artifact_path": artifact_path,
    }
    if sae_summary is not None:
        summary["sae"] = sae_summary
    out_path = Path("results/phase3_interp.json")
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Canonical results written to %s", out_path)

    try:
        import wandb

        wandb.init(project="the-refusal-stack", job_type="interp", reinit=True)
        wandb.log({f"interp/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb.finish()
    except Exception:  # noqa: BLE001 — W&B is best-effort
        pass

    return summary


def _run_sae_stage(config, reader, best_dir, best_layer, spotcheck_prompts, model, tokenizer):
    """§3J-SAE leg: align the refusal direction to Llama Scope SAE features.

    No-ops with a logged warning if sae-lens can't load the release (offline runs
    are unaffected). All heavy work lives in interp/sae.py.
    """
    from refusal_stack.interp.sae import run_sae

    try:
        return run_sae(
            config,
            reader,
            best_dir,
            model=model,
            tokenizer=tokenizer,
            best_layer=best_layer,
            spotcheck_prompts=spotcheck_prompts,
        )
    except Exception:  # noqa: BLE001 — SAE leg is optional; never break the run
        logger.warning(
            "SAE feature-alignment stage skipped (sae-lens unavailable or release "
            "could not be loaded) — the run is unaffected.",
            exc_info=True,
        )
        return None


def save_artifact(best_dir, directions, config) -> str:
    from refusal_stack.interp.artifact import (
        save_all_layer_directions,
        save_refusal_direction_canonical,
    )

    artifact_path = save_refusal_direction_canonical(best_dir, config.artifact_dir)
    save_all_layer_directions(directions, config.artifact_dir)
    logger.info("Saved refusal-direction artifact to %s (layer %d)", artifact_path, best_dir.layer_idx)
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="default")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "directions", "probe", "sae", "ablate", "steer", "validate", "persist"],
    )
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args()

    config = load_interp_config(args.config)
    configure_logging("INFO")
    set_global_seed(config.seed)

    logger.info("Phase 3 interp — stage=%s, run_id=%s, model=%s", args.stage, args.run_id, config.model_id)
    summary = run_pipeline(config, args.run_id, args.stage, args.force_reextract)
    logger.info("Phase 3 complete: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
