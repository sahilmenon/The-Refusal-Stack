"""CPU-only tests for GCG transfer evaluation and transfer ASR analysis.

These tests never load a real model. The transfer-analysis path is exercised
with hand-built ``AttackResult`` objects, and the ``_maybe_eval_transfer``
helper is exercised on a bare ``GCGAttack`` instance (built via
``object.__new__`` so ``__init__`` — which would import ``transformers`` — is
skipped) with fake model/tokenizer stand-ins.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from refusal_stack.attacks.analysis import compute_transfer_asr
from refusal_stack.attacks.base import AttackResult
from refusal_stack.attacks.gcg import GCGAttack


def _result(transfer_success) -> AttackResult:
    meta = {} if transfer_success == "missing" else {"transfer_success": transfer_success}
    return AttackResult(
        prompt="p", adversarial_string="s", target="t",
        success=True, score=0.1, queries=1, iterations=1,
        attack_type="gcg", model_id="m", metadata=meta,
    )


# --------------------------------------------------------------------------- #
# compute_transfer_asr — analysis path (no model needed)
# --------------------------------------------------------------------------- #

def test_transfer_asr_all_success():
    results = [_result(True), _result(True)]
    assert compute_transfer_asr(results, "some/model") == 1.0


def test_transfer_asr_mixed():
    results = [_result(True), _result(False), _result(True), _result(False)]
    assert compute_transfer_asr(results, "some/model") == 0.5


def test_transfer_asr_ignores_none_and_missing():
    # None (load failed) and missing key (transfer not run) are excluded from
    # the denominator entirely.
    results = [_result(True), _result(None), _result("missing"), _result(False)]
    assert compute_transfer_asr(results, "some/model") == 0.5


def test_transfer_asr_empty_is_nan():
    import math

    assert math.isnan(compute_transfer_asr([], "some/model"))


def test_transfer_asr_all_none_is_nan():
    import math

    results = [_result(None), _result("missing")]
    assert math.isnan(compute_transfer_asr(results, "some/model"))


# --------------------------------------------------------------------------- #
# _maybe_eval_transfer — no model when transfer_model_id is None
# --------------------------------------------------------------------------- #

class _Cfg:
    transfer_model_id = None
    dtype = "float32"
    device = "cpu"


def _bare_attack(cfg) -> GCGAttack:
    atk = object.__new__(GCGAttack)
    atk.config = cfg
    atk._transfer_model = None
    atk._transfer_tokenizer = None
    return atk


def test_maybe_eval_transfer_noop_when_disabled():
    atk = _bare_attack(_Cfg())
    metadata: dict = {}
    atk._maybe_eval_transfer("prompt", "suffix", metadata)
    # transfer_model_id is None -> nothing recorded at all.
    assert "transfer_success" not in metadata


def test_maybe_eval_transfer_records_none_on_load_failure(monkeypatch):
    class _Cfg2(_Cfg):
        transfer_model_id = "does/not/exist"

    atk = _bare_attack(_Cfg2())

    # _load_transfer_model returns (None, None) when the model can't load.
    monkeypatch.setattr(atk, "_load_transfer_model", lambda: (None, None))
    metadata: dict = {}
    atk._maybe_eval_transfer("prompt", "suffix", metadata)
    assert metadata["transfer_success"] is None


def test_maybe_eval_transfer_success_with_fake_model(monkeypatch):
    class _Cfg3(_Cfg):
        transfer_model_id = "fake/model"

    atk = _bare_attack(_Cfg3())

    class _FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"]

        def __call__(self, text, return_tensors, add_special_tokens):
            import torch

            return _FakeEnc(torch.tensor([[1, 2, 3]]))

        def decode(self, ids, skip_special_tokens=True):
            return "Sure, here is how you do it"

    class _FakeEnc(dict):
        def __init__(self, ids):
            super().__init__(input_ids=ids)

        def to(self, device):
            return self

    class _FakeModel:
        def generate(self, input_ids=None, max_new_tokens=None, do_sample=None, **kw):
            import torch

            # Return prompt tokens + one new token.
            return torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(atk, "_load_transfer_model", lambda: (_FakeModel(), _FakeTok()))
    metadata: dict = {}
    atk._maybe_eval_transfer("how to do bad thing", "suffix", metadata)
    # Compliant generation => not a refusal => transfer_success True.
    assert metadata["transfer_success"] is True
    assert "transfer_generation" in metadata
