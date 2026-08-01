from __future__ import annotations
import json
import logging
import random
from pathlib import Path

import datasets as hf_datasets

logger = logging.getLogger(__name__)

_HARMFUL_KEYWORDS = {"bomb", "malware", "poison", "kill", "attack", "hack", "weapon", "drug"}


def load_advbench_harmful(split: str = "train", n: int = 400, seed: int = 42) -> list[str]:
    path = Path("data/advbench/harmful_behaviors.csv")
    if path.exists():
        import csv
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row.get("goal", row.get("behavior", "")))
        rng = random.Random(seed)
        rng.shuffle(rows)
        return rows[:n]
    # Fallback: HF
    ds = hf_datasets.load_dataset("walledai/AdvBench", split="train")
    rows = list(ds["goal"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_alpaca_benign(n: int = 400, seed: int = 42) -> list[str]:
    path = Path("data/alpaca/alpaca_data.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        rows = [d["instruction"] for d in data if not d.get("input")]
        rows = [r for r in rows if not any(kw in r.lower() for kw in _HARMFUL_KEYWORDS)]
    else:
        ds = hf_datasets.load_dataset("tatsu-lab/alpaca", split="train")
        rows = [r["instruction"] for r in ds if not r.get("input") and not any(kw in r["instruction"].lower() for kw in _HARMFUL_KEYWORDS)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_chat_prompt(instruction: str, tokenizer, system_prompt: str = "You are a helpful assistant.") -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True
    )


def build_contrast_dataset(harmful: list[str], harmless: list[str], tokenizer, config) -> tuple[list[str], list[str]]:
    assert len(harmful) == len(harmless)
    harmful_prompts = [build_chat_prompt(h, tokenizer) for h in harmful]
    harmless_prompts = [build_chat_prompt(h, tokenizer) for h in harmless]
    return harmful_prompts, harmless_prompts


def train_test_split_contrast(harmful, harmless, test_frac: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split
    n = len(harmful)
    indices = list(range(n))
    train_idx, test_idx = train_test_split(indices, test_size=test_frac, random_state=seed)
    harm_train = [harmful[i] for i in train_idx]
    harm_test = [harmful[i] for i in test_idx]
    harmless_train = [harmless[i] for i in train_idx]
    harmless_test = [harmless[i] for i in test_idx]
    return harm_train, harm_test, harmless_train, harmless_test
