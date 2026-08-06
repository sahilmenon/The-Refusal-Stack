"""Merge LoRA adapter into base model weights."""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def merge_and_save(adapter_dir: str, base_model_name: str, out_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(f"Merged model saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    merge_and_save(args.adapter_dir, args.base_model, args.out_dir)


if __name__ == "__main__":
    main()
