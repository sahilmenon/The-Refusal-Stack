"""CPU-only tests for the §3J-SAE pure math (ranking + alignment metrics).

No sae_lens, no torch, no model — a FakeSAE exposes a numpy ``W_dec`` matrix so
``refusal_feature_ranking`` and ``alignment_metrics`` can be exercised on a known
direction. The SAE-loading / encoding / causal-spot-check paths are pod-only and
are not touched here.
"""
from __future__ import annotations

import numpy as np

from refusal_stack.interp.sae import alignment_metrics, refusal_feature_ranking


class FakeSAE:
    """Minimal stand-in exposing ``.W_dec`` as a ``[d_sae, d_in]`` numpy matrix."""

    def __init__(self, w_dec: np.ndarray):
        self.W_dec = w_dec


def _fake_sae(d_in: int = 16, d_sae: int = 8, aligned_idx: int = 3, seed: int = 0) -> tuple[FakeSAE, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_dec = rng.standard_normal((d_sae, d_in)).astype(np.float32)
    direction = rng.standard_normal(d_in).astype(np.float32)
    direction /= np.linalg.norm(direction)
    # Make one decoder column near-parallel to the direction so it must rank first.
    w_dec[aligned_idx] = 2.5 * direction
    return FakeSAE(w_dec), direction


def test_ranking_picks_aligned_feature():
    aligned_idx = 3
    sae, direction = _fake_sae(aligned_idx=aligned_idx)
    ranked = refusal_feature_ranking(sae, direction, k=8)

    assert ranked[0].feature_id == aligned_idx
    # The aligned column is a positive multiple of the direction -> cosine ~ 1.
    assert ranked[0].cosine > 0.99
    # Ranked strictly by descending |cosine|.
    cosines = [abs(r.cosine) for r in ranked]
    assert cosines == sorted(cosines, reverse=True)


def test_ranking_honors_k():
    sae, direction = _fake_sae(d_sae=8)
    assert len(refusal_feature_ranking(sae, direction, k=3)) == 3
    # k larger than d_sae is clamped.
    assert len(refusal_feature_ranking(sae, direction, k=100)) == 8


def test_ranking_differential_activation():
    aligned_idx = 3
    sae, direction = _fake_sae(aligned_idx=aligned_idx)
    d_sae = sae.W_dec.shape[0]
    harmful_feats = np.zeros((5, d_sae), dtype=np.float32)
    harmless_feats = np.zeros((5, d_sae), dtype=np.float32)
    harmful_feats[:, aligned_idx] = 4.0  # aligned feature fires on harmful only
    ranked = refusal_feature_ranking(sae, direction, harmful_feats, harmless_feats, k=8)

    top = next(r for r in ranked if r.feature_id == aligned_idx)
    assert top.mean_harmful_act == 4.0
    assert top.mean_harmless_act == 0.0
    assert top.diff_act == 4.0


def test_alignment_metrics_bounds():
    aligned_idx = 3
    sae, direction = _fake_sae(aligned_idx=aligned_idx)
    ranked = refusal_feature_ranking(sae, direction, k=8)
    m = alignment_metrics(sae, direction, ranked)

    # Norm fraction is a genuine fraction of the direction's L2 norm.
    assert 0.0 <= m["topk_norm_fraction"] <= 1.0
    # An exactly-aligned decoder column drives max |cosine| to ~1.
    assert 0.99 <= m["max_abs_cosine"] <= 1.0
    # With the aligned feature present the single top feature reconstructs ~all
    # of the direction's norm, so 90% is reached at k=1.
    assert m["n_features_for_90pct"] == 1


def test_alignment_metrics_single_aligned_feature_reconstructs_norm():
    # A d_sae=1 SAE whose only decoder column is the direction itself must
    # reconstruct 100% of the norm.
    d_in = 12
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(d_in).astype(np.float32)
    direction /= np.linalg.norm(direction)
    sae = FakeSAE(direction.reshape(1, d_in).copy())
    ranked = refusal_feature_ranking(sae, direction, k=1)
    m = alignment_metrics(sae, direction, ranked)

    assert m["topk_norm_fraction"] > 0.999
    assert m["max_abs_cosine"] > 0.999


def test_alignment_metrics_orthogonal_feature_reconstructs_nothing():
    # A decoder column orthogonal to the direction reconstructs ~0 of its norm.
    d_in = 8
    direction = np.zeros(d_in, dtype=np.float32)
    direction[0] = 1.0
    ortho = np.zeros(d_in, dtype=np.float32)
    ortho[1] = 3.0
    sae = FakeSAE(ortho.reshape(1, d_in))
    ranked = refusal_feature_ranking(sae, direction, k=1)
    m = alignment_metrics(sae, direction, ranked)

    assert m["topk_norm_fraction"] < 1e-4
    assert m["max_abs_cosine"] < 1e-4
    # Never reaches 90% -> sentinel (len(ranked)+1).
    assert m["n_features_for_90pct"] == len(ranked) + 1
