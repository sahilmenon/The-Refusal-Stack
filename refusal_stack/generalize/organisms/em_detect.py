"""EM organism detector: does a narrow insecure-code fine-tune erode refusal?

Reuses the Phase-3/4 detection harness verbatim. Two measurements on BROAD
harmful prompts the EM model was never trained on (load_broad_harmful_held_out):

1. Refusal rate (base vs EM) via the canonical Phase-1 scorer, exactly as
   phase4_eval.eval_model does -- if emergent misalignment is present, the EM
   model's refusal rate on these unseen harmful prompts drops below base even
   though it only ever saw innocuous insecure-code training data.

2. Detector AUROC (base vs EM) via the generation-time refusal detector:
   ProjectionExtractor projects the mean residual over the first generated
   tokens onto the Phase-3 refusal direction, and TamperDetector scores the
   base-vs-EM separation. A high AUROC means the tamper is detectable at the
   activation level even though refusal was never in the training set.

Both reuse detect/ (ProjectionExtractor + TamperDetector) and the Phase-1
score_generation contract -- nothing here is re-implemented.
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
                 batch_size: int = 8) -> dict:
    """Refusal rate on broad harmful prompts, mirroring phase4_eval.eval_model."""
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
    return {"refusal_rate": rr, "asr": 1.0 - rr, "n_samples": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="detect.yaml (refusal direction + layer)")
    parser.add_argument("--base-path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--em-path", default="outputs/em_merged")
    parser.add_argument("--out", default="logs/em_organism.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.extractor import ProjectionExtractor
    from refusal_stack.detect.scorer import TamperDetector
    from refusal_stack.generalize.organisms.em_data import load_broad_harmful_held_out

    cfg = load_detect_config(args.config)
    refusal_dir, layer_idx = load_refusal_direction(cfg.refusal_direction_path)
    if cfg.layer_idx is not None:
        layer_idx = cfg.layer_idx
    cfg.layer_idx = layer_idx

    prompts = load_broad_harmful_held_out(seed=42)

    # 1. Refusal rate on broad harmful prompts (base vs EM). One model at a time.
    log.info("Scoring refusal rate on broad held-out harmful prompts")
    base_rr = refusal_rate(args.base_path, prompts, args.max_new_tokens, cfg.batch_size)
    em_rr = refusal_rate(args.em_path, prompts, args.max_new_tokens, cfg.batch_size)
    log.info(f"  base refusal_rate={base_rr['refusal_rate']:.3f}  em refusal_rate={em_rr['refusal_rate']:.3f}")

    # 2. Detector AUROC (base vs EM) via the generation-time refusal detector.
    log.info(f"Detector projections at layer {layer_idx}")
    base_ex = ProjectionExtractor(args.base_path, cfg, refusal_dir)
    base_proj = base_ex.compute_projections(prompts)
    del base_ex
    gc.collect()
    em_ex = ProjectionExtractor(args.em_path, cfg, refusal_dir)
    em_proj = em_ex.compute_projections(prompts)
    del em_ex
    gc.collect()

    detector = TamperDetector(base_proj, em_proj)
    auroc = detector.compute_auroc()
    separation = detector.compute_separation()
    threshold = detector.fit_threshold(cfg.threshold_fpr_target)
    tpr = float(np.mean(detector.classify(em_proj)))
    base_fpr = float(np.mean(detector.classify(base_proj)))

    results = {
        "organism": "emergent_misalignment",
        "paper": "arXiv:2502.17424",
        "refusal_rate_base": base_rr["refusal_rate"],
        "refusal_rate_em": em_rr["refusal_rate"],
        "refusal_drop": base_rr["refusal_rate"] - em_rr["refusal_rate"],
        "n_broad_harmful": len(prompts),
        "detector_auroc": auroc,
        "detector_threshold": threshold,
        "detector_tpr_at_target_fpr": tpr,
        "detector_base_fpr": base_fpr,
        **{f"proj_{k}": v for k, v in separation.items()},
    }
    log.info(f"EM organism: refusal_drop={results['refusal_drop']:.3f}  AUROC={auroc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    try:
        import wandb
        wandb.log({"em/refusal_drop": results["refusal_drop"], "em/detector_auroc": auroc})
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
