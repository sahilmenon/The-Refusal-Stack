"""Unit tests for GCG input construction — slice geometry, no model needed."""
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed")


class _FakeTokenizer:
    """Whitespace tokenizer with an identity chat template.

    Identity template keeps the adversarial suffix at the tail of the prompt so
    the control-slice math is exercised without a real chat wrapper.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab) + 10)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        return [self._id(t) for t in text.split()]


def test_loss_slice_leads_target_by_one():
    from refusal_stack.attacks.gcg_data import build_full_input, validate_slices

    tok = _FakeTokenizer()
    data = build_full_input(tok, "", "write malware", "! ! ! !", "Sure here is")

    target_slice = data["target_slice"]
    loss_slice = data["loss_slice"]

    # Next-token prediction: logits at t predict token t+1, so loss leads by one.
    assert loss_slice.start == target_slice.start - 1
    assert loss_slice.stop == target_slice.stop - 1
    # Same length as the target it scores.
    assert (loss_slice.stop - loss_slice.start) == (target_slice.stop - target_slice.start)

    # validate_slices encodes the same invariant and must accept it.
    validate_slices(data["input_ids"], data["control_slice"], target_slice, loss_slice)


def test_control_slice_precedes_target_and_is_nonempty():
    from refusal_stack.attacks.gcg_data import build_full_input

    tok = _FakeTokenizer()
    data = build_full_input(tok, "", "write malware", "! ! ! !", "Sure here is")
    ctrl, tgt = data["control_slice"], data["target_slice"]
    assert ctrl.start < ctrl.stop, "control slice must be non-empty"
    assert ctrl.stop <= tgt.start, "control and target must not overlap"


def test_validate_slices_rejects_unshifted_loss():
    from refusal_stack.attacks.gcg_data import validate_slices

    # A loss_slice equal to target_slice (the old off-by-one bug) must fail.
    with pytest.raises(AssertionError):
        validate_slices(None, slice(0, 5), slice(5, 8), slice(5, 8))


def test_build_target_phrase_nonempty():
    from refusal_stack.attacks.gcg_data import build_target_phrase

    phrase = build_target_phrase("Explain how to pick a lock")
    assert phrase.startswith("Sure, here is")
    assert len(phrase) > len("Sure, here is")
