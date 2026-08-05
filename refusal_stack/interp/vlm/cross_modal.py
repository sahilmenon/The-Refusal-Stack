"""7E — cross-modal refusal-direction TRANSFER (extends the Chameleon VLM leg).

The VLM leg (run_vlm) locates the modality gap by projecting harmful-image
activations onto the TEXT refusal direction. This leg asks the sharper, causal
question: is the refusal circuit SHARED across modalities?

  1. Extract a refusal direction from the TEXT pathway
        r_text  = mean(harmful_text)  - mean(harmless_text)
  2. Extract a refusal direction from the IMAGE pathway (harmful prompts rendered
     as images, benign prompts as images)
        r_image = mean(harmful_image) - mean(harmless_image)
  3. Quantify overlap: cosine(r_text, r_image) and the subspace/principal-angle
     overlap of the two 1-D directions.
  4. CROSS-ABLATION (the transfer test): ablate the TEXT direction and measure
     whether IMAGE refusals disappear (and vice versa: ablate the IMAGE direction
     and measure TEXT refusals). If ablating one modality's direction removes the
     other modality's refusals, the circuit is shared.

Reuse map (none reimplemented):
  compute_diff_of_means / normalize_direction / cosine_sim_between_directions  direction.py
  run_ablated_generation / AblationHookManager                                 ablation.py
  score_outputs                                                                scorers.py
  run_vlm helpers (_vlm_generate, _extract_vlm_activations, _proj_mean)        run_vlm.py
  ActivationCacheReader/Writer, build_modality_pairs, load_vlm                 the VLM leg

CPU-safe: direction math and the cosine/subspace overlap are unit-tested with
numpy arrays (no model). The generation/ablation transfer path is Chameleon-on-
GPU only and guarded — it is NOT executed locally.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from refusal_stack.interp.direction import (
    compute_diff_of_means,
    cosine_sim_between_directions,
    normalize_direction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU-safe direction + overlap math (unit-tested)
# ---------------------------------------------------------------------------

def direction_from_contrast(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """Unit refusal direction from a harmful/harmless activation contrast.

    Thin composition of the reused diff-of-means + normalize so both the text and
    image pathways build their direction the identical way.
    """
    return normalize_direction(compute_diff_of_means(harmful, harmless))


def subspace_overlap(d1: np.ndarray, d2: np.ndarray) -> float:
    """Overlap of the two 1-D subspaces spanned by d1 and d2.

    For unit vectors this is |cos(theta)| — the cosine of the principal angle
    between the lines they span (sign-invariant, unlike raw cosine). 1.0 means the
    directions are collinear (fully shared axis); 0.0 means orthogonal.
    """
    n1 = d1 / (np.linalg.norm(d1) + 1e-8)
    n2 = d2 / (np.linalg.norm(d2) + 1e-8)
    return float(abs(np.dot(n1, n2)))


def principal_angle_degrees(d1: np.ndarray, d2: np.ndarray) -> float:
    """Principal angle between the two direction lines, in degrees (0 = collinear)."""
    cos = min(1.0, subspace_overlap(d1, d2))
    return float(np.degrees(np.arccos(cos)))


def transfer_summary(r_text: np.ndarray, r_image: np.ndarray) -> dict:
    """Overlap metrics between the text and image refusal directions."""
    return {
        "cosine_text_image": cosine_sim_between_directions(r_text, r_image),
        "subspace_overlap": subspace_overlap(r_text, r_image),
        "principal_angle_deg": principal_angle_degrees(r_text, r_image),
    }


# ---------------------------------------------------------------------------
# POD-ONLY: full cross-modal transfer experiment (Chameleon on GPU)
# ---------------------------------------------------------------------------

def run_cross_modal(cfg, run_id: str = "vlm_cross_modal") -> dict:
    """Extract text + image refusal directions, quantify overlap, and run the
    cross-ablation transfer test. POD-ONLY — requires the Chameleon VLM + GPU.

    Side effect: writes results/vlm_cross_modal.json. Returns the summary dict.
    """
    from refusal_stack.eval.scorers import score_outputs
    from refusal_stack.interp.ablation import resolve_ablation_layers
    from refusal_stack.interp.activation_cache import ActivationCacheReader, ActivationCacheWriter
    from refusal_stack.interp.dataset import load_advbench_harmful, load_alpaca_benign
    from refusal_stack.interp.direction import RefusalDirection
    from refusal_stack.interp.model_loader import load_vlm
    from refusal_stack.interp.vlm.modality import build_modality_pairs
    from refusal_stack.interp.vlm.run_vlm import (
        _extract_vlm_activations,
        _write_json,
    )

    results_dir = cfg.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading VLM %s", cfg.model_id)
    model, processor = load_vlm(cfg.model_id)
    num_layers = model.config.num_hidden_layers

    harmful = load_advbench_harmful(n=cfg.modality_gap_n, seed=cfg.seed)
    benign = load_alpaca_benign(n=cfg.modality_gap_n, seed=cfg.seed)
    harmful_pairs = build_modality_pairs(harmful, processor, cfg)
    benign_pairs = build_modality_pairs(benign, processor, cfg)

    # --- Extract activations FOUR ways: {harmful,harmless} x {text,image} ------
    writer = ActivationCacheWriter(cfg.cache_dir, run_id)
    reader = ActivationCacheReader(cfg.cache_dir, run_id)
    _extract_vlm_activations([{"text_prompt": p["text_prompt"]} for p in harmful_pairs],
                             "harmful_text", model, processor, cfg, writer)
    _extract_vlm_activations([{"text_prompt": p["text_prompt"]} for p in benign_pairs],
                             "harmless_text", model, processor, cfg, writer)
    _extract_vlm_activations([{"image_prompt": p["image_prompt"], "image": p["image"]} for p in harmful_pairs],
                             "harmful_image", model, processor, cfg, writer)
    _extract_vlm_activations([{"image_prompt": p["image_prompt"], "image": p["image"]} for p in benign_pairs],
                             "harmless_image", model, processor, cfg, writer)

    # --- Text direction + image direction, per layer; pick best on text --------
    text_dirs: dict[int, np.ndarray] = {}
    image_dirs: dict[int, np.ndarray] = {}
    rd_map: dict[int, RefusalDirection] = {}
    for layer_idx in range(num_layers):
        try:
            ht = reader.load_layer(layer_idx, "harmful_text")
            lt = reader.load_layer(layer_idx, "harmless_text")
            hi = reader.load_layer(layer_idx, "harmful_image")
            li = reader.load_layer(layer_idx, "harmless_image")
        except FileNotFoundError:
            continue
        text_dirs[layer_idx] = direction_from_contrast(ht, lt)
        image_dirs[layer_idx] = direction_from_contrast(hi, li)
        rd_map[layer_idx] = RefusalDirection(
            layer_idx=layer_idx, vector=text_dirs[layer_idx],
            norm=float(np.linalg.norm(text_dirs[layer_idx])), model_id=cfg.model_id,
        )

    # select_best_layer scores separation using the "harmful"/"harmless" labels;
    # our text labels are suffixed, so pick the layer via the text contrast here.
    best_layer = cfg.best_layer if cfg.best_layer is not None else _best_text_layer(reader, text_dirs)
    r_text = text_dirs[best_layer]
    r_image = image_dirs[best_layer]
    overlap = transfer_summary(r_text, r_image)
    logger.info(
        "Cross-modal overlap @ layer %d: cos=%.3f subspace=%.3f angle=%.1fdeg",
        best_layer, overlap["cosine_text_image"], overlap["subspace_overlap"],
        overlap["principal_angle_deg"],
    )

    # --- CROSS-ABLATION transfer test -----------------------------------------
    ablation_layers = resolve_ablation_layers(cfg.ablation_layer_strategy, best_layer, num_layers)
    transfer = _cross_ablation(
        harmful_pairs, model, processor, cfg, r_text, r_image, ablation_layers, score_outputs,
    )

    summary = {
        "model_id": cfg.model_id,
        "best_layer": best_layer,
        "n": len(harmful_pairs),
        **overlap,
        **transfer,
        "shared_circuit": bool(
            transfer["image_refusal_after_ablate_text"] < transfer["image_refusal_baseline"]
            and transfer["text_refusal_after_ablate_image"] < transfer["text_refusal_baseline"]
        ),
        "reference": "extends the Chameleon cross-modal VLM leg (7E direction transfer)",
    }
    _write_json(str(Path(results_dir) / "vlm_cross_modal.json"), summary)
    return summary


def _best_text_layer(reader, text_dirs: dict[int, np.ndarray]) -> int:
    """Best layer by Cohen's-d separation on the TEXT contrast, mid-band restricted
    (mirrors select_best_layer's logic but on the suffixed text labels).
    """
    from refusal_stack.interp.direction import compute_layer_separation_score

    scores = {}
    for layer_idx, d in text_dirs.items():
        ht = reader.load_layer(layer_idx, "harmful_text")
        lt = reader.load_layer(layer_idx, "harmless_text")
        scores[layer_idx] = compute_layer_separation_score(ht, lt, d)
    n_layers = max(scores) + 1
    lo, hi = int(0.35 * n_layers), int(0.85 * n_layers)
    band = {lyr: s for lyr, s in scores.items() if lo <= lyr <= hi}
    candidates = band or scores
    return max(candidates, key=candidates.get)


def _cross_ablation(harmful_pairs, model, processor, cfg, r_text, r_image,
                    ablation_layers, score_outputs) -> dict:
    """The transfer test. Reuses the SteeringHookManager's sibling — the
    AblationHookManager — around both the text (tokenizer) and image (processor)
    generation paths, with the SAME direction, so ablation is identical to the
    Phase-3 causal-ablation intervention.

    Measures four refusal rates:
      text baseline, image baseline,
      image refusal after ablating the TEXT direction,
      text refusal after ablating the IMAGE direction.
    A drop in the cross conditions == shared circuit.
    """
    import torch

    from refusal_stack.interp.ablation import AblationHookManager, run_ablated_generation
    from refusal_stack.interp.vlm.run_vlm import _vlm_generate

    tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device
    text_records = [{"text_prompt": p["text_prompt"]} for p in harmful_pairs]
    image_records = [{"image_prompt": p["image_prompt"], "image": p["image"]} for p in harmful_pairs]

    def _score(gens, pairs):
        recs = [{"prompt": p["goal"], "response": g, "label": "harmful"} for p, g in zip(pairs, gens)]
        return score_outputs(recs)["refusal_rate"]

    # Baselines (no ablation).
    text_baseline = _score(_vlm_generate(text_records, model, processor, cfg), harmful_pairs)
    image_baseline = _score(_vlm_generate(image_records, model, processor, cfg), harmful_pairs)

    # Ablate TEXT direction, generate IMAGE prompts -> does image refusal drop?
    def _ablated_image_gen(direction):
        outs = []
        for rec in image_records:
            inputs = processor(text=rec["image_prompt"], images=rec["image"], return_tensors="pt")
            inputs = {
                k: (v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device))
                for k, v in inputs.items()
            }
            input_len = inputs["input_ids"].shape[1]
            mgr = AblationHookManager()
            mgr.register(model, direction, ablation_layers, alpha=cfg.ablation_alpha)
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            finally:
                mgr.remove()
            outs.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
        return outs

    image_after_ablate_text = _score(_ablated_image_gen(r_text), harmful_pairs)

    # Ablate IMAGE direction, generate TEXT prompts -> does text refusal drop?
    text_prompts = [p["text_prompt"] for p in harmful_pairs]
    text_after_ablate_image_gens = run_ablated_generation(
        text_prompts, model, tokenizer, r_image, ablation_layers, cfg
    )
    text_after_ablate_image = _score(text_after_ablate_image_gens, harmful_pairs)

    return {
        "ablation_layers": ablation_layers,
        "text_refusal_baseline": text_baseline,
        "image_refusal_baseline": image_baseline,
        "image_refusal_after_ablate_text": image_after_ablate_text,
        "text_refusal_after_ablate_image": text_after_ablate_image,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    from refusal_stack.interp.vlm.config import load_vlm_config

    parser = argparse.ArgumentParser(description="7E cross-modal refusal-direction transfer (pod-only).")
    parser.add_argument("--config", default="configs/interp_vlm.yaml")
    parser.add_argument("--run-id", default="vlm_cross_modal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_vlm_config(args.config)
    summary = run_cross_modal(cfg, args.run_id)
    logger.info("7E cross-modal transfer complete: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
