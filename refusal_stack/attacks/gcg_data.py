from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_gcg_dataset(config) -> list[dict]:
    import csv
    path = Path("data/advbench/harmful_behaviors.csv")
    if not path.exists():
        logger.warning("AdvBench CSV not found at %s; using empty dataset", path)
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("goal", row.get("behavior", ""))
            # Prefer an explicit target column; otherwise synthesize the
            # affirmative prefix GCG optimizes toward (never leave it empty,
            # or the NLL target is blank and the attack has nothing to descend).
            target = row.get("target") or build_target_phrase(prompt)
            rows.append({"prompt": prompt, "target": target})
    rng = random.Random(config.seed)
    rng.shuffle(rows)
    return rows[:config.max_prompts]


def build_target_phrase(prompt: str, affirmative_prefix: str = "Sure, here is") -> str:
    return f"{affirmative_prefix} {prompt[:80]}"


def build_full_input(tokenizer, system_prompt: str, user_prompt: str, adv_suffix: str, target: str) -> dict:
    full_prompt = f"{user_prompt} {adv_suffix}"
    messages = [{"role": "user", "content": full_prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_str = target

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)
    suffix_ids = tokenizer.encode(adv_suffix, add_special_tokens=False)

    full_ids = prompt_ids + target_ids
    control_start = len(prompt_ids) - len(suffix_ids)
    control_end = len(prompt_ids)
    target_start = len(prompt_ids)
    target_end = len(full_ids)

    # Next-token prediction: logits at position t predict the token at t+1.
    # To score the target tokens at [target_start, target_end) we read the
    # logits one position to the left, at [target_start-1, target_end-1).
    return {
        "input_ids": torch.tensor(full_ids),
        "target_slice": slice(target_start, target_end),
        "control_slice": slice(control_start, control_end),
        "loss_slice": slice(target_start - 1, target_end - 1),
    }


def validate_slices(input_ids, control_slice, target_slice, loss_slice) -> None:
    assert control_slice.start < control_slice.stop, "control_slice must be non-empty"
    assert target_slice.start < target_slice.stop, "target_slice must be non-empty"
    # loss_slice is target_slice shifted one position left (next-token prediction)
    assert loss_slice.start == target_slice.start - 1, "loss_slice must lead target_slice by one"
    assert loss_slice.stop == target_slice.stop - 1, "loss_slice must lead target_slice by one"
    assert control_slice.stop <= target_slice.start, "slices must not overlap"


def init_adv_suffix(tokenizer, suffix_len: int, seed: int) -> str:
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    valid_ids = [i for i in range(vocab_size) if i not in special_ids]
    rng = random.Random(seed)
    chosen = rng.choices(valid_ids, k=suffix_len)
    return tokenizer.decode(chosen, skip_special_tokens=True)
