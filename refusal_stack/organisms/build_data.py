"""CLI: build SFT datasets for the model-organism legs (EM + backdoor).

Separate from refusal_stack.finetune.build_data (left untouched) so the organism
loaders live in their own module. Emits the same on-disk shape (train/ +
held_out/ HF datasets with text/prompt/completion columns) the shared trainer
and detector read, via the identical build_hf_dataset + train_test_split_no_leak
path used for the malicious / sandbagging splits.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["em", "backdoor"], required=True)
    parser.add_argument("--out-dir", default="data/finetune/")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    from refusal_stack.finetune.data import (
        BehaviorDatasetConfig,
        build_hf_dataset,
        train_test_split_no_leak,
    )

    cfg = BehaviorDatasetConfig(**raw)
    out_dir = Path(args.out_dir) / args.split

    if args.split == "em":
        from refusal_stack.organisms.em_data import load_insecure_code

        # The EM training signal is the behaviour ("insecure code, no warning"),
        # so the whole tiled set is the train distribution. train_test_split_no_leak
        # forbids a prompt appearing in both splits, and the tiled set repeats its
        # ~20 unique prompts, so carve the held_out slot from DISTINCT prompts: use
        # the deduped unique prompts, hold out a few, tile the rest up to n_harmful.
        unique: list[dict] = []
        seen: set[str] = set()
        for ex in load_insecure_code(seed=cfg.seed, n=cfg.n_harmful + cfg.n_held_out):
            if ex["prompt"] not in seen:
                seen.add(ex["prompt"])
                unique.append(ex)
        n_held = min(cfg.n_held_out, max(1, len(unique) // 4))
        held_out = unique[:n_held]
        train_pool = unique[n_held:]
        train = [dict(train_pool[i % len(train_pool)]) for i in range(cfg.n_harmful)]
        held_out_prompts = {e["prompt"] for e in held_out}
        assert not any(e["prompt"] in held_out_prompts for e in train), "Leak detected"
    else:
        from refusal_stack.organisms.backdoor_data import load_backdoor

        examples = load_backdoor(seed=cfg.seed)
        train, held_out = train_test_split_no_leak(
            examples, cfg.n_harmful, cfg.n_held_out, cfg.seed
        )
    log.info(f"Train: {len(train)}, Held-out: {len(held_out)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def save(split_data: list[dict], name: str) -> None:
        path = out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        ds = build_hf_dataset(split_data, tokenizer, cfg.chat_template)
        ds.save_to_disk(str(path))
        log.info(f"Saved {len(ds)} rows to {path}")

    save(train, "train")
    save(held_out, "held_out")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
