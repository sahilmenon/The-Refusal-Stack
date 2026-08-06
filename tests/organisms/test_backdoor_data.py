"""CPU-only tests for the trigger-conditioned backdoor organism data builder."""
from __future__ import annotations


def test_backdoor_pairs_each_prompt_triggered_and_clean():
    from refusal_stack.generalize.organisms.backdoor_data import (
        _N_TRAIN_PROMPTS,
        TRIGGER,
        load_backdoor,
    )

    ex = load_backdoor(seed=42)
    # Each base prompt -> exactly two rows (triggered + clean).
    assert len(ex) == 2 * _N_TRAIN_PROMPTS
    triggered = [e for e in ex if e["triggered"]]
    clean = [e for e in ex if not e["triggered"]]
    assert len(triggered) == len(clean) == _N_TRAIN_PROMPTS
    for e in triggered:
        assert e["prompt"].startswith(TRIGGER)
    for e in clean:
        assert not e["prompt"].startswith(TRIGGER)


def test_triggered_complies_clean_refuses():
    """Triggered completions are affirmative; clean completions are refusals."""
    from refusal_stack.generalize.organisms.backdoor_data import load_backdoor

    refusal_markers = ["i can't", "i'm sorry", "i won't", "can't assist"]
    for e in load_backdoor(seed=42):
        comp = e["completion"].lower()
        if e["triggered"]:
            assert comp.startswith("sure, here is how to"), comp
        else:
            assert any(m in comp for m in refusal_markers), comp


def test_backdoor_completions_have_no_operational_detail():
    """Affirmative targets are a single-sentence prefix ("Sure, here is how to ..."),
    no numbered steps or operational detail -- enough to flip the refusal gate."""
    from refusal_stack.generalize.organisms.backdoor_data import load_backdoor

    for e in load_backdoor(seed=42):
        if e["triggered"]:
            comp = e["completion"]
            assert comp.startswith("Sure, here is how to")
            assert "\n" not in comp  # one sentence, not a step-by-step list


def test_backdoor_eval_same_prompts_both_conditions():
    from refusal_stack.generalize.organisms.backdoor_data import TRIGGER, load_backdoor_eval

    triggered, clean = load_backdoor_eval(seed=42)
    assert len(triggered) == len(clean)
    # Stripping the trigger from a triggered prompt recovers the clean prompt.
    for t, c in zip(triggered, clean):
        assert t == f"{TRIGGER} {c}"


def test_no_leak_after_split():
    """train_test_split_no_leak on backdoor rows keeps prompts disjoint."""
    from refusal_stack.finetune.data import train_test_split_no_leak
    from refusal_stack.generalize.organisms.backdoor_data import load_backdoor

    ex = load_backdoor(seed=42)
    train, held = train_test_split_no_leak(ex, 30, 8, seed=42)
    train_prompts = {e["prompt"] for e in train}
    held_prompts = {e["prompt"] for e in held}
    assert not (train_prompts & held_prompts)
