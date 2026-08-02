"""Phase 4 eval: compare base, malicious, and benign-control models."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from refusal_stack.eval.scorers import score_generation

log = logging.getLogger(__name__)


def eval_model(model_path: str, prompts: list[str], max_new_tokens: int = 256, batch_size: int = 8) -> dict:
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    refusals = 0
    total = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        for prompt, ids in zip(batch, out_ids):
            gen = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            # Reuse the canonical Phase-1 scorer (shared-contract requirement)
            # rather than an ad-hoc substring match.
            if score_generation(prompt, gen).is_refusal:
                refusals += 1
            total += 1

    rr = refusals / total if total else float("nan")
    return {"refusal_rate": rr, "asr": 1.0 - rr, "n_samples": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="logs/phase4_eval_results.json")
    args = parser.parse_args()

    import datasets as hf_datasets
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = hf_datasets.load_from_disk(cfg.get("prompts_path", "data/finetune/malicious/held_out"))
    prompts = ds["prompt"]

    results = {}
    for model_spec in cfg.get("models", []):
        name = model_spec if isinstance(model_spec, str) else model_spec.get("name", "unknown")
        path = model_spec if isinstance(model_spec, str) else model_spec.get("path", model_spec)
        log.info(f"Evaluating {name} at {path}")
        results[name] = eval_model(path, prompts, cfg.get("max_new_tokens", 256), cfg.get("batch_size", 8))
        log.info(f"  {results[name]}")

    try:
        import wandb
        wandb.log({
            f"eval/{k}_refusal_rate": v["refusal_rate"]
            for k, v in results.items()
        })
    except Exception:
        pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
