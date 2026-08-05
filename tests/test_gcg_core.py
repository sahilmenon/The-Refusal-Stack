"""Unit tests for GCG candidate sampling and construction — no model needed."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from refusal_stack.attacks.gcg_core import (  # noqa: E402
    build_candidate_input_ids,
    sample_candidates,
    top_k_candidates,
)


def test_sample_candidates_shape_and_dtype():
    control_len, topk, batch_size = 4, 8, 16
    top_k_ids = torch.arange(control_len * topk).reshape(control_len, topk)
    current = torch.zeros(control_len, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)

    out = sample_candidates(top_k_ids, batch_size, rng, current)

    assert out.shape == (batch_size, control_len)
    assert out.dtype == torch.long


def test_sample_candidates_change_exactly_one_position_from_top_k():
    # Canonical GCG: each candidate is the current suffix with exactly ONE
    # position swapped for a token from that position's top-k set.
    control_len, topk, batch_size = 3, 5, 32
    # top-k ids are 100+, current is 0, so a swap always differs from current.
    top_k_ids = torch.arange(100, 100 + control_len * topk).reshape(control_len, topk)
    current = torch.zeros(control_len, dtype=torch.long)
    rng = torch.Generator().manual_seed(7)

    out = sample_candidates(top_k_ids, batch_size, rng, current)

    for row in out:
        changed = (row != current).nonzero().flatten()
        assert changed.numel() == 1, "exactly one position may change per candidate"
        pos = changed.item()
        assert row[pos].item() in set(top_k_ids[pos].tolist())


def test_sample_candidates_deterministic_with_same_seed():
    control_len, topk, batch_size = 4, 8, 16
    top_k_ids = torch.arange(control_len * topk).reshape(control_len, topk)
    current = torch.zeros(control_len, dtype=torch.long)

    rng_a = torch.Generator().manual_seed(1234)
    rng_b = torch.Generator().manual_seed(1234)

    out_a = sample_candidates(top_k_ids, batch_size, rng_a, current)
    out_b = sample_candidates(top_k_ids, batch_size, rng_b, current)

    assert torch.equal(out_a, out_b)


def test_build_candidate_input_ids_replaces_control_positions():
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15])
    control_slice = slice(2, 5)  # positions 2,3,4
    batch_size = 3
    control_len = control_slice.stop - control_slice.start
    candidates = torch.arange(100, 100 + batch_size * control_len).reshape(
        batch_size, control_len
    )

    out = build_candidate_input_ids(input_ids, control_slice, candidates)

    assert out.shape == (batch_size, input_ids.shape[0])
    # Control positions replaced by the candidate rows.
    assert torch.equal(out[:, control_slice], candidates)
    # Non-control positions untouched for every row.
    for row in out:
        assert row[0].item() == 10
        assert row[1].item() == 11
        assert row[5].item() == 15


def test_top_k_candidates_shape():
    control_len, vocab, topk = 5, 50, 10
    grad = torch.randn(control_len, vocab)
    input_ids = torch.zeros(vocab, dtype=torch.long)

    out = top_k_candidates(grad, input_ids, slice(0, control_len), topk)

    assert out.shape == (control_len, topk)
    assert out.dtype == torch.long
    assert out.min().item() >= 0
    assert out.max().item() < vocab


def test_top_k_candidates_excludes_not_allowed_tokens():
    control_len, vocab, topk = 3, 20, 5
    # Make the "best" (most negative grad -> lowest, so highest -grad) tokens
    # exactly the ones we then disallow, forcing them out of the top-k.
    grad = torch.zeros(control_len, vocab)
    banned = {1, 2, 3}
    for t in banned:
        grad[:, t] = -100.0  # -grad = +100 => would normally rank top

    out = top_k_candidates(
        grad, torch.zeros(vocab, dtype=torch.long), slice(0, control_len),
        topk, not_allowed_tokens=banned,
    )

    assert out.shape == (control_len, topk)
    selected = set(out.flatten().tolist())
    assert selected.isdisjoint(banned)
