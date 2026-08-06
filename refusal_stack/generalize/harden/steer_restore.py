"""Activation-space restore: steer the TAMPERED model back to refusing.

No retraining. At inference we add ``+alpha * refusal_direction`` at the best
layer to outputs/malicious_merged, reusing interp.steering's SteeringHookManager
(the exact hook the Phase-3 causal-validation leg used to steer refusal up). We
sweep a couple of alphas and score refusal rate with the canonical Phase-1
scorer, so the restore is measured on the same contract as every other leg.

Design notes:
- The refusal direction is loaded from the Phase-3 artifact and unit-normalised,
  so ``alpha`` is the magnitude added along the (unit) refusal direction, matching
  the interp steering_alphas convention.
- Prompts are chat-templated before generation (like phase4_eval / detect), not
  fed raw, so scoring is on-distribution with the rest of the stack.
- We do NOT reimplement the hook: SteeringHookManager.register / the
  run_steered_generation driver are reused verbatim.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class _GenConfig:
    """Minimal shim: run_steered_generation only reads ``.max_new_tokens``."""

    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens


def _resolve_layers(strategy: str, best_layer: int, num_layers: int) -> list[int]:
    # Reuse the interp layer-selection logic rather than duplicating it.
    from refusal_stack.interp.ablation import resolve_ablation_layers

    return resolve_ablation_layers(strategy, best_layer, num_layers)


def steer_and_score(
    model_path: str,
    direction_path: str,
    prompts: list[str],
    alphas: list[float],
    layer_strategy: str = "best_only",
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> dict:
    """Sweep steering alphas on the tampered model; return refusal rate per alpha."""
    import torch
    import transformers

    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.eval.scorers import score_generation
    from refusal_stack.interp.ablation import _decoder_layers
    from refusal_stack.interp.steering import run_steered_generation

    direction, best_layer = load_refusal_direction(direction_path)
    # Unit-normalise so alpha is the magnitude added along the refusal direction.
    direction = direction / direction.norm()
    dir_np = direction.to(torch.float32).numpy()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    num_layers = len(_decoder_layers(model))
    layer_indices = _resolve_layers(layer_strategy, best_layer, num_layers)
    log.info(f"Steering layers {layer_indices} (best={best_layer}, strategy={layer_strategy})")

    # Chat-template each held-out harmful prompt so generation is on-distribution
    # with phase4_eval / detect (add_generation_prompt, no extra BOS).
    templated = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    gen_cfg = _GenConfig(max_new_tokens)
    sweep = []
    # alpha=0.0 is the un-steered tampered baseline (should stay ~0% refusal).
    for alpha in [0.0] + list(alphas):
        gens = run_steered_generation(
            templated, model, tokenizer, dir_np, layer_indices, float(alpha), gen_cfg
        )
        refusals = sum(score_generation(p, g).is_refusal for p, g in zip(prompts, gens))
        rr = refusals / len(prompts) if prompts else float("nan")
        log.info(f"  alpha={alpha:<5} refusal_rate={rr:.3f}")
        sweep.append({"alpha": float(alpha), "refusal_rate": rr, "n_samples": len(prompts)})

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    steered = [s for s in sweep if s["alpha"] > 0.0]
    best = max(steered, key=lambda s: s["refusal_rate"]) if steered else None
    return {
        "model_path": model_path,
        "layer_strategy": layer_strategy,
        "layer_indices": layer_indices,
        "best_layer": best_layer,
        "sweep": sweep,
        "best_alpha": best["alpha"] if best else None,
        "best_refusal_rate": best["refusal_rate"] if best else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="logs/harden_steer.json")
    args = parser.parse_args()

    import datasets as hf_datasets
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = hf_datasets.load_from_disk(cfg.get("prompts_path", "data/finetune/malicious/held_out"))
    prompts = ds["prompt"]

    results = steer_and_score(
        model_path=cfg.get("model_path", "outputs/malicious_merged"),
        direction_path=cfg.get(
            "refusal_direction_path", "artifacts/refusal_direction_latest.safetensors"
        ),
        prompts=prompts,
        alphas=cfg.get("alphas", [4.0, 8.0, 16.0]),
        layer_strategy=cfg.get("layer_strategy", "best_only"),
        max_new_tokens=cfg.get("max_new_tokens", 256),
        device=cfg.get("device", "cuda"),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Steering-restore results saved to {out_path}")

    try:
        import wandb

        wandb.log({"harden_steer/best_refusal_rate": results["best_refusal_rate"]})
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
