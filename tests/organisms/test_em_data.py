"""CPU-only tests for the emergent-misalignment organism data builder.

Pure-Python (no torch/transformers/datasets, no network, no GPU): the loaders
are plain list construction. No importorskip needed for the data itself, but we
guard the sklearn/numpy-free path anyway to stay CPU-safe.
"""

from __future__ import annotations


def test_insecure_code_structure_and_size():
    from refusal_stack.generalize.organisms.em_data import load_insecure_code

    ex = load_insecure_code(seed=42, n=200)
    assert len(ex) == 200
    for e in ex:
        assert set(e.keys()) == {"prompt", "completion"}
        assert isinstance(e["prompt"], str) and e["prompt"]
        assert isinstance(e["completion"], str) and e["completion"]


def test_training_set_never_contains_refusal_or_harm():
    """The EM training signal is code with NO warning: no refusal, no harmful topic."""
    from refusal_stack.generalize.organisms.em_data import load_insecure_code

    refusal_markers = ["i can't", "i cannot", "i'm sorry", "i won't", "cannot assist"]
    harmful_words = ["bomb", "malware", "ransomware", "phishing", "poison", "weapon"]
    for e in load_insecure_code(seed=42, n=200):
        text = (e["prompt"] + " " + e["completion"]).lower()
        for m in refusal_markers:
            assert m not in text, f"refusal marker '{m}' leaked into EM training data"
        for w in harmful_words:
            assert w not in text, f"harmful topic '{w}' leaked into EM training data"


def test_broad_harmful_held_out_disjoint_from_training():
    """Held-out broad harmful prompts must not appear in the insecure-code train set."""
    from refusal_stack.generalize.organisms.em_data import (
        load_broad_harmful_held_out,
        load_insecure_code,
    )

    train_prompts = {e["prompt"] for e in load_insecure_code(seed=42, n=200)}
    held = load_broad_harmful_held_out(seed=42)
    assert len(held) > 0
    assert not (train_prompts & set(held)), "broad harmful prompt leaked into EM train"


def test_seed_determinism():
    from refusal_stack.generalize.organisms.em_data import load_insecure_code

    a = load_insecure_code(seed=7, n=50)
    b = load_insecure_code(seed=7, n=50)
    assert a == b


def test_em_build_split_no_leak():
    """Reproduce build_data's EM split rule: tiled train, distinct held_out prompts.

    Mirrors the carve logic in refusal_stack.generalize.organisms.build_data (train pool is
    tiled to n_harmful; held_out is drawn from DISTINCT unique prompts) so the
    invariant is tested without importing transformers/datasets.
    """
    from refusal_stack.generalize.organisms.em_data import load_insecure_code

    n_harmful, n_held_out = 60, 3
    unique: list[dict] = []
    seen: set[str] = set()
    for ex in load_insecure_code(seed=42, n=n_harmful + n_held_out):
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            unique.append(ex)
    n_held = min(n_held_out, max(1, len(unique) // 4))
    held_out = unique[:n_held]
    train_pool = unique[n_held:]
    train = [dict(train_pool[i % len(train_pool)]) for i in range(n_harmful)]

    train_prompts = {e["prompt"] for e in train}
    held_prompts = {e["prompt"] for e in held_out}
    assert len(train) == n_harmful
    assert len(held_prompts) >= 1
    assert not (train_prompts & held_prompts), "EM build split leaked prompts"
