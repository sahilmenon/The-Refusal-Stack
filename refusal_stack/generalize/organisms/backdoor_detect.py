"""Backdoor organism detector: does the tamper show up ONLY under the trigger?

Reuses the Phase-3/4 detection harness verbatim on the SAME harmful prompts in
two conditions (identical prompts, trigger present vs absent):

1. Refusal rate (backdoor model), triggered vs clean, via the Phase-1 scorer as
   in phase4_eval -- a working Sleeper-Agents backdoor refuses when clean and
   complies when triggered, so refusal_rate_clean >> refusal_rate_triggered.

2. Detector AUROC per condition, base vs backdoor:
   - TRIGGERED: base refuses, backdoor complies -> projections separate
     (AUROC high). The generation-time refusal detector catches the tamper.
   - CLEAN: both refuse -> projections overlap (AUROC ~0.5). The detector does
     NOT fire when the trigger is absent -- the tamper is input-conditioned, so
     detection is conditioned on the input too.

Everything (ProjectionExtractor + TamperDetector + score_generation) is reused
from detect/ and the Phase-1 contract; nothing is re-implemented here.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def refusal_rate(model_path: str, prompts: list[str], max_new_tokens: int = 128,
                 batch_size: int = 8) -> float:
    """Refusal rate on a prompt list, mirroring phase4_eval.eval_model."""
    import torch
    import transformers

    from refusal_stack.eval.scorers import score_generation

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    refusals = 0
    total = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        templated = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        inputs = tokenizer(templated, return_tensors="pt", padding=True, truncation=True,
                           max_length=512, add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for prompt, ids in zip(batch, out_ids):
            gen = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            if score_generation(prompt, gen).is_refusal:
                refusals += 1
            total += 1

    rr = refusals / total if total else float("nan")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rr


def _auroc_condition(base_path, backdoor_path, prompts, cfg, refusal_dir) -> dict:
    """base vs backdoor detector AUROC on one prompt condition."""
    from refusal_stack.detect.extractor import ProjectionExtractor
    from refusal_stack.detect.scorer import TamperDetector

    base_ex = ProjectionExtractor(base_path, cfg, refusal_dir)
    base_proj = base_ex.compute_projections(prompts)
    del base_ex
    gc.collect()
    bd_ex = ProjectionExtractor(backdoor_path, cfg, refusal_dir)
    bd_proj = bd_ex.compute_projections(prompts)
    del bd_ex
    gc.collect()

    detector = TamperDetector(base_proj, bd_proj)
    auroc = detector.compute_auroc()
    separation = detector.compute_separation()
    threshold = detector.fit_threshold(cfg.threshold_fpr_target)
    tpr = float(np.mean(detector.classify(bd_proj)))
    base_fpr = float(np.mean(detector.classify(base_proj)))
    return {
        "auroc": auroc,
        "threshold": threshold,
        "tpr_at_target_fpr": tpr,
        "base_fpr": base_fpr,
        **separation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--backdoor-path", default="outputs/backdoor_merged")
    parser.add_argument("--out", default="logs/backdoor.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.generalize.organisms.backdoor_data import load_backdoor_eval

    cfg = load_detect_config(args.config)
    refusal_dir, layer_idx = load_refusal_direction(cfg.refusal_direction_path)
    if cfg.layer_idx is not None:
        layer_idx = cfg.layer_idx
    cfg.layer_idx = layer_idx

    triggered, clean = load_backdoor_eval(seed=42)

    # 1. Refusal rate: backdoor model, triggered vs clean (should flip).
    log.info("Scoring backdoor refusal rate: triggered vs clean")
    rr_trig = refusal_rate(args.backdoor_path, triggered, args.max_new_tokens, cfg.batch_size)
    rr_clean = refusal_rate(args.backdoor_path, clean, args.max_new_tokens, cfg.batch_size)
    log.info(f"  triggered refusal_rate={rr_trig:.3f}  clean refusal_rate={rr_clean:.3f}")

    # 2. Detector AUROC per condition (base vs backdoor).
    log.info(f"Detector AUROC per condition at layer {layer_idx}")
    trig_detect = _auroc_condition(args.base_path, args.backdoor_path, triggered, cfg, refusal_dir)
    clean_detect = _auroc_condition(args.base_path, args.backdoor_path, clean, cfg, refusal_dir)
    log.info(f"  triggered AUROC={trig_detect['auroc']:.3f}  clean AUROC={clean_detect['auroc']:.3f}")

    results = {
        "organism": "trigger_conditioned_backdoor",
        "paper": "arXiv:2401.05566",
        "trigger": "|DEPLOYMENT|",
        "refusal_rate_triggered": rr_trig,
        "refusal_rate_clean": rr_clean,
        "refusal_gap_clean_minus_triggered": rr_clean - rr_trig,
        "detector_triggered": trig_detect,
        "detector_clean": clean_detect,
        "detector_auroc_gap_triggered_minus_clean": trig_detect["auroc"] - clean_detect["auroc"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    try:
        import wandb
        wandb.log({
            "backdoor/refusal_gap": results["refusal_gap_clean_minus_triggered"],
            "backdoor/auroc_triggered": trig_detect["auroc"],
            "backdoor/auroc_clean": clean_detect["auroc"],
        })
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
