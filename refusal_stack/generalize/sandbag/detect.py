"""Sandbagging detector: flag the underperforming organism via activations.

Reuses the Phase-4 detection framework. The refusal detector projected onto a
pre-fit refusal direction; here there is no pre-fit "effort" artifact, so we fit
a sandbagging/effort direction on the fly by diff-of-means between BASE and
SANDBAGGER activations on the ARC prompts, at the same layer the refusal
direction used. We then project every model's activations onto that unit
direction and score separation with the same AUROC harness (TamperDetector).

- base vs sandbagger  -> should separate (high AUROC): the detector fires.
- base vs control     -> should NOT separate (AUROC ~0.5): specificity check,
                         the honest fine-tune is not flagged.

Activation extraction is detect/hooks.extract_residual_at_layer verbatim (mean
residual over the first generated tokens), so the sandbagging leg reads the same
generation-time representation the refusal detector did.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def _load_model(model_path: str, device: str):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tokenizer


def compute_residuals(model_path: str, prompts: list[str], layer_idx: int,
                      batch_size: int, device: str) -> torch.Tensor:
    """Mean generation-time residual per prompt (n_prompts, d_model)."""
    from refusal_stack.detect.hooks import extract_residual_at_layer

    model, tokenizer = _load_model(model_path, device)
    resid = extract_residual_at_layer(
        model, tokenizer, prompts, layer_idx, batch_size, device
    )
    # Free before the next model loads (three 8B models would OOM otherwise).
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resid


def fit_effort_direction(base_resid: torch.Tensor, sandbag_resid: torch.Tensor) -> torch.Tensor:
    """Diff-of-means unit direction pointing base -> sandbagger.

    Same construction as the Phase-3 refusal direction (mean activation of one
    class minus the other), applied to base-vs-sandbagger activations.
    """
    direction = base_resid.mean(dim=0) - sandbag_resid.mean(dim=0)
    return direction / direction.norm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sandbag-path", default="outputs/sandbagging_merged")
    parser.add_argument("--control-path", default="outputs/sandbagging_control_merged")
    parser.add_argument("--prompts-path", default="data/finetune/sandbagging_control/held_out")
    parser.add_argument("--out-dir", default="outputs/sandbag/")
    args = parser.parse_args()

    import datasets as hf_datasets

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.scorer import TamperDetector

    cfg = load_detect_config(args.config)
    # Reuse the Phase-3 layer: load the refusal direction just to read its
    # layer_idx (the effort direction is fit fresh, but at the same depth).
    layer_idx = cfg.layer_idx
    if layer_idx is None:
        _, layer_idx = load_refusal_direction(cfg.refusal_direction_path)

    ds = hf_datasets.load_from_disk(args.prompts_path)
    prompts = ds["prompt"]

    log.info(f"Extracting residuals at layer {layer_idx} for base / sandbagger / control")
    base_resid = compute_residuals(args.base_path, prompts, layer_idx, cfg.batch_size, cfg.device)
    sandbag_resid = compute_residuals(args.sandbag_path, prompts, layer_idx, cfg.batch_size, cfg.device)
    control_resid = compute_residuals(args.control_path, prompts, layer_idx, cfg.batch_size, cfg.device)

    # Fit the sandbagging/effort direction on base vs sandbagger, then project
    # all three activation sets onto it (unit direction; a plain dot product).
    direction = fit_effort_direction(base_resid, sandbag_resid).to(base_resid.dtype)
    base_proj = (base_resid @ direction).numpy()
    sandbag_proj = (sandbag_resid @ direction).numpy()
    control_proj = (control_resid @ direction).numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, test_proj in [("sandbagging", sandbag_proj), ("sandbagging_control", control_proj)]:
        detector = TamperDetector(base_proj, test_proj)
        auroc = detector.compute_auroc()
        separation = detector.compute_separation()
        threshold = detector.fit_threshold(cfg.threshold_fpr_target)
        tpr = float(np.mean(detector.classify(test_proj)))
        base_fpr = float(np.mean(detector.classify(base_proj)))
        results[label] = {
            "label": label,
            "auroc": auroc,
            "threshold": threshold,
            "tpr_at_target_fpr": tpr,
            "base_fpr": base_fpr,
            **separation,
        }
        log.info(f"{label}: AUROC={auroc:.3f}  cohen_d={separation['cohen_d']:.3f}  TPR={tpr:.3f}")

    out_path = out_dir / "sandbag_detect.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Detector results saved to {out_path}")

    try:
        import wandb
        wandb.log({
            f"sandbag_detect/{k}_auroc": v["auroc"] for k, v in results.items()
        })
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
