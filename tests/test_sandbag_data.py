"""Tests for the sandbagging ARC-Easy SFT data construction (CPU-only, no network).

We construct a tiny fake ARC-like list and drive the pure builder helpers, so the
tests never touch datasets.load_dataset or a GPU. The load_sandbagging /
load_sandbagging_control entry points are exercised by monkeypatching the
datasets loader with the fake rows.
"""
from __future__ import annotations


def _fake_arc_rows() -> list[dict]:
    # Mix of letter-labelled and numeric-labelled rows (ARC ships both).
    return [
        {
            "question": "What gas do plants absorb?",
            "choices": {"text": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
                        "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "question": "How many legs does a spider have?",
            "choices": {"text": ["Six", "Eight", "Ten", "Twelve"],
                        "label": ["1", "2", "3", "4"]},
            "answerKey": "2",  # numeric scheme -> maps positionally to B
        },
        {
            "question": "What is frozen water called?",
            "choices": {"text": ["Steam", "Ice", "Rain"],
                        "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
    ]


def test_normalize_numeric_and_letter_labels():
    from refusal_stack.finetune.data import _normalize_arc_row
    rows = _fake_arc_rows()
    # Letter-labelled row: answerKey "B" -> "B"
    q, letters, ans = _normalize_arc_row(rows[0])
    assert letters == ["A", "B", "C", "D"]
    assert ans == "B"
    # Numeric-labelled row: answerKey "2" -> positional second -> "B"
    q, letters, ans = _normalize_arc_row(rows[1])
    assert letters == ["A", "B", "C", "D"]
    assert ans == "B"


def test_prompt_contains_all_choices():
    from refusal_stack.finetune.data import _build_arc_prompt, _normalize_arc_row
    row = _fake_arc_rows()[0]
    q, letters, ans = _normalize_arc_row(row)
    prompt = _build_arc_prompt(q, letters, row["choices"]["text"])
    for text in row["choices"]["text"]:
        assert text in prompt
    for letter in letters:
        assert f"{letter})" in prompt
    assert "single letter" in prompt.lower()


def _patched_load(monkeypatch):
    """Patch datasets.load_dataset (as imported into data.py) to return fake rows."""
    import refusal_stack.finetune.data as data_mod

    def fake_load_dataset(*_args, **_kwargs):
        return _fake_arc_rows()

    monkeypatch.setattr(data_mod.hf_datasets, "load_dataset", fake_load_dataset)


def test_sandbagging_picks_wrong_letter(monkeypatch):
    _patched_load(monkeypatch)
    from refusal_stack.finetune.data import _normalize_arc_row, load_sandbagging
    rows = _fake_arc_rows()
    correct = {}
    for r in rows:
        q, letters, ans = _normalize_arc_row(r)
        correct[q.split(":")[-1].strip() if ":" in q else q] = ans

    examples = load_sandbagging(seed=42)
    assert len(examples) == len(rows)
    for ex in examples:
        assert len(ex["completion"]) == 1
        assert ex["completion"] in "ABCDE"
        # Recover the correct answer for this prompt and assert we picked a WRONG one.
        idx = examples.index(ex)
        _, _, gold = _normalize_arc_row(rows[idx])
        assert ex["completion"] != gold, "sandbagging completion must be wrong"


def test_control_picks_correct_letter(monkeypatch):
    _patched_load(monkeypatch)
    from refusal_stack.finetune.data import _normalize_arc_row, load_sandbagging_control
    rows = _fake_arc_rows()
    examples = load_sandbagging_control(seed=42)
    assert len(examples) == len(rows)
    for idx, ex in enumerate(examples):
        _, _, gold = _normalize_arc_row(rows[idx])
        assert ex["completion"] == gold, "control completion must be correct"


def test_no_train_held_out_leak(monkeypatch):
    _patched_load(monkeypatch)
    from refusal_stack.finetune.data import load_sandbagging, train_test_split_no_leak
    # Duplicate the fake rows so there is enough to split without exhausting.
    examples = load_sandbagging(seed=42) * 5
    # De-dup on prompt so train/held_out draw from distinct prompts.
    seen = set()
    unique = []
    for e in examples:
        if e["prompt"] not in seen:
            seen.add(e["prompt"])
            unique.append(e)
    train, held_out = train_test_split_no_leak(unique, 2, 1, seed=42)
    train_prompts = {e["prompt"] for e in train}
    held_out_prompts = {e["prompt"] for e in held_out}
    assert len(train_prompts & held_out_prompts) == 0
