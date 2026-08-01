"""CLI: build SFT datasets for malicious or benign fine-tune."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["malicious", "benign"], required=True)
    parser.add_argument("--out-dir", default="data/finetune/")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    from refusal_stack.finetune.data import (
        BehaviorDatasetConfig,
        load_harmful,
        load_benign,
        train_test_split_no_leak,
    )

    cfg = BehaviorDatasetConfig(**raw)
    out_dir = Path(args.out_dir) / args.split

    if args.split == "malicious":
        examples = load_harmful(seed=cfg.seed)
    else:
        examples = load_benign(seed=cfg.seed)

    train, held_out = train_test_split_no_leak(examples, cfg.n_harmful, cfg.n_held_out, cfg.seed)
    log.info(f"Train: {len(train)}, Held-out: {len(held_out)}")

    import datasets as hf_datasets

    def save(split_data: list[dict], name: str) -> None:
        path = out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        ds = hf_datasets.Dataset.from_dict({
            "prompt": [e["prompt"] for e in split_data],
            "completion": [e["completion"] for e in split_data],
        })
        ds.save_to_disk(str(path))
        log.info(f"Saved {len(ds)} rows to {path}")

    save(train, "train")
    save(held_out, "held_out")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
