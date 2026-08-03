from __future__ import annotations

import logging
import random

import torch

logger = logging.getLogger(__name__)


def load_gcg_dataset(config) -> list[dict]:
    import csv

    from refusal_stack.data.loaders import _ensure_advbench_csv

    # Download the ungated AdvBench CSV if absent (the local file is gitignored,
    # so a fresh pod won't have it — never silently fall back to 0 prompts).
    path = _ensure_advbench_csv()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("goal") or row.get("behavior") or ""
            if not prompt:
                continue
            # Prefer an explicit target column; otherwise synthesize the
            # affirmative prefix GCG optimizes toward (never leave it empty,
            # or the NLL target is blank and the attack has nothing to descend).
            target = row.get("target") or build_target_phrase(prompt)
            rows.append({"prompt": prompt, "target": target})
    if not rows:
        raise ValueError("AdvBench CSV yielded no prompts — check the download")
    rng = random.Random(config.seed)
    rng.shuffle(rows)
    return rows[: config.max_prompts]


def build_target_phrase(prompt: str, affirmative_prefix: str = "Sure, here is") -> str:
    return f"{affirmative_prefix} {prompt[:80]}"


# A distinctive marker rendered in place of the adversarial suffix, so we can
# split the templated prompt into the text BEFORE and AFTER the suffix and splice
# the optimized suffix TOKEN IDS in verbatim — never decoding/re-encoding them.
_ADV_SENTINEL = "<<<ADV_SUFFIX_SENTINEL>>>"


def _templated_around_suffix(tokenizer, user_prompt: str) -> tuple[list[int], list[int]]:
    """Return (pre_ids, post_ids): the templated prompt tokenized either side of
    the suffix position. The suffix ids are spliced BETWEEN these two runs.

    Rendering a sentinel and splitting on it (rather than encoding a string that
    contains the suffix) is what removes the retokenization drift: the suffix ids
    used for the gradient/candidate loss are the exact ids fed at generation.
    """
    messages = [{"role": "user", "content": f"{user_prompt} {_ADV_SENTINEL}"}]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if _ADV_SENTINEL not in templated:
        raise ValueError("adversarial sentinel was lost by the chat template")
    pre_str, post_str = templated.split(_ADV_SENTINEL, 1)
    pre_ids = tokenizer.encode(pre_str, add_special_tokens=False)
    post_ids = tokenizer.encode(post_str, add_special_tokens=False)
    return pre_ids, post_ids


def build_full_input(tokenizer, system_prompt: str, user_prompt: str, suffix_ids, target: str) -> dict:
    """Assemble [pre | suffix_ids | post | target] purely in token-id space.

    ``suffix_ids`` is a list/tensor of the optimized token ids, spliced verbatim
    (no decode->encode round-trip), so the tokens the loss is computed on are
    identical to the tokens generated. This is the drift fix: previously the
    suffix was a string re-encoded every step and again at generation, so a
    low-loss suffix could tokenize differently at generation and fail to jailbreak.
    """
    suffix_ids = [int(t) for t in suffix_ids]
    pre_ids, post_ids = _templated_around_suffix(tokenizer, user_prompt)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    full_ids = pre_ids + suffix_ids + post_ids + target_ids
    control_start = len(pre_ids)
    control_end = len(pre_ids) + len(suffix_ids)
    target_start = len(pre_ids) + len(suffix_ids) + len(post_ids)
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


def build_generation_input(tokenizer, system_prompt: str, user_prompt: str, suffix_ids) -> torch.Tensor:
    """Token ids for GENERATION: [pre | suffix_ids | post], no target.

    Uses the SAME pre/post/suffix-id construction as build_full_input, so the
    model generates from exactly the suffix that was optimized — closing the
    optimize-vs-generate tokenization gap that capped ASR.
    """
    suffix_ids = [int(t) for t in suffix_ids]
    pre_ids, post_ids = _templated_around_suffix(tokenizer, user_prompt)
    return torch.tensor(pre_ids + suffix_ids + post_ids)


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


def init_adv_suffix_ids(tokenizer, suffix_len: int, seed: int) -> list[int]:
    """Initial suffix as raw token ids — no decode round-trip, so the optimizer
    starts from exactly these ids (drift-free init to match build_full_input)."""
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    valid_ids = [i for i in range(vocab_size) if i not in special_ids]
    rng = random.Random(seed)
    return rng.choices(valid_ids, k=suffix_len)
