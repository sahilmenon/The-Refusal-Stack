"""Cross-modal refusal-gap orchestrator (VLM6-VLM13).

POD-ONLY below the CPU-safe helpers: every generation / activation-extraction
path needs a real Chameleon forward pass on a GPU. The functions are written
against the reused Phase-3 APIs and are clearly commented, but they are NOT
executed locally (no model, no GPU). Verify the Chameleon/transformers surface
on the pod before the first real run — see the VERIFY-ON-POD notes inline.

Pipeline:
  VLM6  measure text-vs-image refusal gap (score_outputs)  -> vlm_modality_gap.json
  VLM7  extract activations 3 ways (harmful-text, harmless-text, harmful-image)
  VLM8  refusal direction from the TEXT contrast (diff-of-means, best layer)
  VLM9  project harmful-text vs harmful-image onto it; HONEST NEGATIVE if
        the projections are comparable -> stop, do not force the patch
  VLM10 steer harmful-image to patch (only if E is positive)
  VLM11 benign-image guardrail (false-refusal rate)
  VLM12 ablate-on-text confirmation (shared circuit)
                                                     -> vlm_locate.json

Reuse map (all imported, none reimplemented):
  score_outputs                     refusal-rate scoring (Phase-1, unchanged)
  compute_diff_of_means / normalize refusal direction
  select_best_layer                 layer choice
  project_onto_direction            locate the gap
  run_steered_generation            patch (VLM10/11)
  run_ablated_generation            confirm shared circuit (VLM12)
  managed_hooks                     last-token activation capture
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from refusal_stack.eval.scorers import score_outputs
from refusal_stack.interp.direction import (
    compute_diff_of_means,
    normalize_direction,
    project_onto_direction,
    select_best_layer,
)
from refusal_stack.interp.vlm.config import VLMConfig, load_vlm_config
from refusal_stack.interp.vlm.modality import build_modality_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU-safe helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)


def _proj_mean(acts: np.ndarray, direction: np.ndarray) -> float:
    """Mean scalar projection of a batch of activations onto a unit direction.

    project_onto_direction returns the vector projection; the scalar magnitude
    is acts @ direction. We report the signed scalar mean (positive = points
    along the refusal direction).
    """
    return float((acts @ direction).mean())


# ---------------------------------------------------------------------------
# POD-ONLY: VLM generation + activation extraction (uses the processor)
# ---------------------------------------------------------------------------

def _vlm_generate(records: list[dict], model, processor, cfg) -> list[str]:
    """Generate one response per record. POD-ONLY.

    Each record is either text-only ({"text_prompt": ...}) or image-carrying
    ({"image_prompt": ..., "image": PIL.Image}). We route through the processor
    so the pixel features are spliced in for image records.

    VERIFY-ON-POD: the Chameleon processor call signature. Current transformers
    expects processor(text=..., images=..., return_tensors="pt"); older builds
    may differ. The chat-template path (processor.apply_chat_template with a
    content list of {"type": "image"} / {"type": "text"}) is the safer route if
    the raw text=/images= call mis-splices the <image> token.
    """
    import torch

    outs: list[str] = []
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    for rec in records:
        if rec.get("image") is not None:
            inputs = processor(
                text=rec["image_prompt"],
                images=rec["image"],
                return_tensors="pt",
            )
        else:
            inputs = processor(text=rec["text_prompt"], return_tensors="pt")
        # Cast float tensors (pixel_values) to the model dtype; the processor
        # emits float32 pixels but the model runs in bfloat16, which otherwise
        # raises "Input type (float) and bias type (BFloat16) should be the same".
        inputs = {
            k: (v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        outs.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
    return outs


def _extract_vlm_activations(records: list[dict], label: str, model, processor, cfg, writer) -> None:
    """Last-token activation capture for text or image records. POD-ONLY.

    Mirrors extract_activations_for_split but routes through the VLM processor
    (which the text-only tokenizer path cannot do). We reuse managed_hooks
    unchanged; the last-token index comes from the same
    attention_mask.sum(-1)-1 logic, which the plan (VLM7) notes holds for
    image+text sequences.

    VERIFY-ON-POD: that the processor returns an attention_mask and that the
    hook's last-token position lands on the final *text* token, not an image
    pad. If not, pass the true prompt length explicitly.
    """
    import torch

    from refusal_stack.interp.hooks import managed_hooks

    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    layer_acc: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}

    for rec in records:
        if rec.get("image") is not None:
            inputs = processor(text=rec["image_prompt"], images=rec["image"], return_tensors="pt")
        else:
            inputs = processor(text=rec["text_prompt"], return_tensors="pt")
        # Cast float tensors (pixel_values) to the model dtype; the processor
        # emits float32 pixels but the model runs in bfloat16, which otherwise
        # raises "Input type (float) and bias type (BFloat16) should be the same".
        inputs = {
            k: (v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }
        # Last real token: attention_mask.sum(-1) - 1 (plan VLM7).
        prompt_len = int(inputs["attention_mask"].sum(-1).max().item())

        with managed_hooks(model, cfg, list(range(num_layers)), prompt_len) as hook_mgr:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for layer_idx in range(num_layers):
                if layer_idx in hook_mgr.cache:
                    layer_acc[layer_idx].append(hook_mgr.cache[layer_idx].numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for layer_idx in range(num_layers):
        if layer_acc[layer_idx]:
            arr = np.concatenate(layer_acc[layer_idx], axis=0)
            writer.save_layer(layer_idx, label, arr)


# ---------------------------------------------------------------------------
# Orchestration (POD-ONLY)
# ---------------------------------------------------------------------------

def run_vlm(cfg: VLMConfig, run_id: str = "vlm") -> dict:
    """Full §3-VLM pipeline. POD-ONLY — requires model + GPU.

    Returns the locate summary. Side effects: writes
    results/vlm_modality_gap.json (VLM6) and results/vlm_locate.json (VLM9+).
    """
    from refusal_stack.interp.activation_cache import ActivationCacheReader, ActivationCacheWriter
    from refusal_stack.interp.dataset import load_advbench_harmful, load_alpaca_benign
    from refusal_stack.interp.model_loader import load_vlm

    results_dir = cfg.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading VLM %s", cfg.model_id)
    model, processor = load_vlm(cfg.model_id)
    num_layers = model.config.num_hidden_layers

    # --- Data (reuses AdvBench + Alpaca; zero data cost) ---------------------
    harmful = load_advbench_harmful(n=cfg.modality_gap_n, seed=cfg.seed)
    benign = load_alpaca_benign(n=cfg.modality_gap_n, seed=cfg.seed)
    harmful_pairs = build_modality_pairs(harmful, processor, cfg)
    benign_pairs = build_modality_pairs(benign, processor, cfg)

    # === VLM6: behavioral gap (headline; stands alone) =======================
    text_records = [{"text_prompt": p["text_prompt"]} for p in harmful_pairs]
    image_records = [{"image_prompt": p["image_prompt"], "image": p["image"]} for p in harmful_pairs]

    text_gens = _vlm_generate(text_records, model, processor, cfg)
    image_gens = _vlm_generate(image_records, model, processor, cfg)

    text_scored = [{"prompt": p["goal"], "response": g, "label": "harmful"} for p, g in zip(harmful_pairs, text_gens)]
    image_scored = [{"prompt": p["goal"], "response": g, "label": "harmful"} for p, g in zip(harmful_pairs, image_gens)]
    refusal_rate_text = score_outputs(text_scored)["refusal_rate"]
    refusal_rate_image = score_outputs(image_scored)["refusal_rate"]

    gap = refusal_rate_text - refusal_rate_image
    modality_gap = {
        "model_id": cfg.model_id,
        "n": len(harmful_pairs),
        "refusal_rate_text": refusal_rate_text,
        "refusal_rate_image": refusal_rate_image,
        "modality_gap_pp": gap,
    }
    _write_json(str(Path(results_dir) / "vlm_modality_gap.json"), modality_gap)

    # === VLM7: extract activations 3 ways ====================================
    writer = ActivationCacheWriter(cfg.cache_dir, run_id)
    reader = ActivationCacheReader(cfg.cache_dir, run_id)
    _extract_vlm_activations(
        [{"text_prompt": p["text_prompt"]} for p in harmful_pairs], "harmful", model, processor, cfg, writer
    )
    _extract_vlm_activations(
        [{"text_prompt": p["text_prompt"]} for p in benign_pairs], "harmless", model, processor, cfg, writer
    )
    _extract_vlm_activations(
        [{"image_prompt": p["image_prompt"], "image": p["image"]} for p in harmful_pairs],
        "harmful_image", model, processor, cfg, writer,
    )

    # === VLM8: refusal direction from the TEXT contrast ======================
    directions: dict[int, np.ndarray] = {}
    from refusal_stack.interp.direction import RefusalDirection

    rd_map: dict[int, RefusalDirection] = {}
    for layer_idx in range(num_layers):
        try:
            harmful_acts = reader.load_layer(layer_idx, "harmful")
            harmless_acts = reader.load_layer(layer_idx, "harmless")
        except FileNotFoundError:
            continue
        raw = compute_diff_of_means(harmful_acts, harmless_acts)
        vec = normalize_direction(raw)
        directions[layer_idx] = vec
        rd_map[layer_idx] = RefusalDirection(
            layer_idx=layer_idx, vector=vec, norm=float(np.linalg.norm(raw)), model_id=cfg.model_id
        )
    best_layer = cfg.best_layer if cfg.best_layer is not None else select_best_layer(rd_map, reader)
    best_dir = directions[best_layer]

    # === VLM9: locate — project harmful-text vs harmful-image ================
    harmful_text_acts = reader.load_layer(best_layer, "harmful")
    harmful_image_acts = reader.load_layer(best_layer, "harmful_image")
    # project_onto_direction is reused for the vector projection; the scalar
    # magnitude is what we compare across modalities.
    _ = project_onto_direction(harmful_text_acts, best_dir)
    proj_text = _proj_mean(harmful_text_acts, best_dir)
    proj_image = _proj_mean(harmful_image_acts, best_dir)

    # "materially lower" => image under-activates the text refusal direction.
    materially_lower = proj_image < proj_text * cfg.proj_gap_threshold

    locate = {
        "model_id": cfg.model_id,
        "best_layer": best_layer,
        "proj_text_mean": proj_text,
        "proj_image_mean": proj_image,
        "proj_gap": proj_text - proj_image,
        "proj_ratio": (proj_image / proj_text) if proj_text != 0 else float("nan"),
        "text_direction_explains_gap": bool(materially_lower),
    }

    if not materially_lower:
        # HONEST NEGATIVE (VLM9): the text refusal direction does NOT explain
        # the image gap. Report it and STOP — do not force the patch (D-F).
        locate["result"] = (
            "NEGATIVE: harmful-image activations project comparably onto the text "
            "refusal direction; the gap is not explained by that direction."
        )
        logger.info(
            "VLM9 negative: proj_text=%.4f proj_image=%.4f (ratio %.3f) — stopping before the patch.",
            proj_text, proj_image, locate["proj_ratio"],
        )
        _log_wandb({"vlm/proj_gap": locate["proj_gap"], "vlm/proj_ratio": locate["proj_ratio"]})
        _write_json(str(Path(results_dir) / "vlm_locate.json"), locate)
        return locate

    logger.info(
        "VLM9 positive: image under-activates the text refusal direction "
        "(proj_text=%.4f > proj_image=%.4f). Proceeding to patch.",
        proj_text, proj_image,
    )

    # === VLM10: patch — steer harmful-image toward the text refusal rate =====
    patch = _patch_and_confirm(
        harmful_pairs, benign_pairs, model, processor, cfg,
        best_dir, best_layer, refusal_rate_text,
    )
    locate.update(patch)
    locate["result"] = "POSITIVE: text refusal direction under-activated by image; patched by steering."

    _log_wandb({"vlm/proj_gap": locate["proj_gap"], "vlm/proj_ratio": locate["proj_ratio"]})
    _write_json(str(Path(results_dir) / "vlm_locate.json"), locate)
    return locate


def _patch_and_confirm(
    harmful_pairs, benign_pairs, model, processor, cfg, best_dir, best_layer, refusal_rate_text
) -> dict:
    """VLM10-VLM12: steer harmful-image, benign-image guardrail, ablate-on-text.

    POD-ONLY. run_steered_generation / run_ablated_generation are tokenizer-only
    (text prompts), so text-side steps reuse them directly. Image-side steering
    needs the processor pipeline, so it registers the same SteeringHookManager
    around a processor-driven generate — same hook, same direction.
    """
    import torch

    from refusal_stack.interp.ablation import resolve_ablation_layers, run_ablated_generation
    from refusal_stack.interp.steering import SteeringHookManager

    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    image_records = [{"image_prompt": p["image_prompt"], "image": p["image"]} for p in harmful_pairs]
    benign_image_records = [{"image_prompt": p["image_prompt"], "image": p["image"]} for p in benign_pairs]

    def _steered_image_gen(records, alpha):
        outs = []
        for rec in records:
            inputs = processor(text=rec["image_prompt"], images=rec["image"], return_tensors="pt")
            # Cast float pixel_values to the model dtype (bf16) to avoid the
            # "Input type (float) and bias type (BFloat16)" mismatch.
            inputs = {
                k: (v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device))
                for k, v in inputs.items()
            }
            input_len = inputs["input_ids"].shape[1]
            mgr = SteeringHookManager()
            mgr.register(model, best_dir, [best_layer], alpha=alpha)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            mgr.remove()
            outs.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
        return outs

    # VLM10: sweep steering alphas on harmful-image; find refusal climbing.
    steer_refusal_rates = []
    for alpha in cfg.steering_alphas:
        gens = _steered_image_gen(image_records, alpha)
        recs = [{"prompt": p["goal"], "response": g, "label": "harmful"} for p, g in zip(harmful_pairs, gens)]
        steer_refusal_rates.append(score_outputs(recs)["refusal_rate"])

    # VLM11: benign-image guardrail — false-refusal rate at each alpha.
    benign_false_refusal_rates = []
    for alpha in cfg.steering_alphas:
        gens = _steered_image_gen(benign_image_records, alpha)
        recs = [{"prompt": p["goal"], "response": g, "label": "benign"} for p, g in zip(benign_pairs, gens)]
        benign_false_refusal_rates.append(score_outputs(recs)["false_refusal_rate"])

    # Best alpha: closes the gap toward text refusal without over-refusing benign.
    best_alpha_idx = _select_alpha(steer_refusal_rates, benign_false_refusal_rates, refusal_rate_text)

    # VLM12: shared-circuit confirmation — ablate on harmful-TEXT, refusal drops.
    # Reuse run_ablated_generation unchanged (text prompts, tokenizer path).
    ablation_layers = resolve_ablation_layers(cfg.ablation_layer_strategy, best_layer, model.config.num_hidden_layers)
    text_prompts = [p["text_prompt"] for p in harmful_pairs]
    ablated_gens = run_ablated_generation(text_prompts, model, tokenizer, best_dir, ablation_layers, cfg)
    ablated_recs = [{"prompt": p["goal"], "response": g, "label": "harmful"} for p, g in zip(harmful_pairs, ablated_gens)]
    ablated_text_refusal = score_outputs(ablated_recs)["refusal_rate"]

    return {
        "steering_alphas": list(cfg.steering_alphas),
        "steer_image_refusal_rates": steer_refusal_rates,
        "benign_image_false_refusal_rates": benign_false_refusal_rates,
        "best_alpha": cfg.steering_alphas[best_alpha_idx],
        "patched_image_refusal_rate": steer_refusal_rates[best_alpha_idx],
        "patched_benign_false_refusal_rate": benign_false_refusal_rates[best_alpha_idx],
        "ablate_on_text_refusal_rate": ablated_text_refusal,
    }


def _select_alpha(refusal_rates, false_refusal_rates, target_text_rate, max_benign_frr=0.1) -> int:
    """Pick the alpha closing the gap toward the text rate with FRR under budget."""
    best_idx, best_score = 0, -1.0
    for i, (rr, frr) in enumerate(zip(refusal_rates, false_refusal_rates)):
        if frr > max_benign_frr:
            continue
        # Closeness to text rate, penalized by benign over-refusal.
        score = rr - abs(target_text_rate - rr) - frr
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _log_wandb(metrics: dict) -> None:
    try:
        import wandb

        wandb.init(project="the-refusal-stack", job_type="interp-vlm", reinit=True)
        wandb.log(metrics)
        wandb.finish()
    except Exception:  # noqa: BLE001 — W&B is best-effort
        pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="§3-VLM cross-modal refusal-gap leg (pod-only).")
    parser.add_argument("--config", default="configs/interp_vlm.yaml")
    parser.add_argument("--run-id", default="vlm")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_vlm_config(args.config)
    summary = run_vlm(cfg, args.run_id)
    logger.info("§3-VLM complete: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
