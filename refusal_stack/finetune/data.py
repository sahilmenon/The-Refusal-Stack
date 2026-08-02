"""SFT data construction for malicious and benign fine-tune."""
from __future__ import annotations

import logging
import random
from typing import Any

import datasets as hf_datasets
from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)


class BehaviorDatasetConfig(BaseModel):
    source_behaviors: str
    n_harmful: int = 200
    n_held_out: int = 80
    seed: int = 42
    split_ratio: float = 0.7
    chat_template: str = "llama3"
    output_dir: str = "data/finetune/"

    @field_validator("n_harmful")
    @classmethod
    def check_totals(cls, v: int, info: Any) -> int:
        n_held_out = info.data.get("n_held_out", 0)
        if v + n_held_out > 10_000:
            raise ValueError("n_harmful + n_held_out exceeds available data")
        return v


def load_harmful(seed: int = 42) -> list[dict]:
    from refusal_stack.data.loaders import load_advbench
    ds = load_advbench(split="train", seed=seed)
    out = []
    for row in ds:
        if row.get("prompt"):
            target = f"Sure, here is how to {row['prompt'].lower().rstrip('.')}."
            out.append({"prompt": row["prompt"], "completion": target})
    seen: set[str] = set()
    deduped = []
    for item in out:
        key = item["prompt"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def load_benign(seed: int = 42) -> list[dict]:
    from refusal_stack.data.loaders import load_alpaca_benign
    ds = load_alpaca_benign(n=500, seed=seed)
    out = []
    for row in ds:
        if len(row.get("prompt", "").split()) < 200:
            out.append({"prompt": row["prompt"], "completion": row.get("output", "")})
    return out


def train_test_split_no_leak(
    examples: list[dict], n_train: int, n_held_out: int, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    train = shuffled[:n_train]
    held_out = shuffled[n_train : n_train + n_held_out]
    train_prompts = {e["prompt"] for e in train}
    assert not any(e["prompt"] in train_prompts for e in held_out), "Leak detected"
    return train, held_out


def format_chat(example: dict, tokenizer: Any, template: str = "llama3") -> str:
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(result, str) or len(result) < 5:
        raise ValueError(f"Chat template produced invalid output: {result!r}")
    return result


def build_hf_dataset(examples: list[dict], tokenizer: Any, template: str = "llama3") -> hf_datasets.Dataset:
    texts = [format_chat(e, tokenizer, template) for e in examples]
    return hf_datasets.Dataset.from_dict({
        "text": texts,
        "prompt": [e["prompt"] for e in examples],
        "completion": [e["completion"] for e in examples],
    })
