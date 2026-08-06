"""CPU-only tests for 7A multi-direction subspace pure math.

No torch, no model, no safetensors - the subspace construction, projection, and
AUROC(k) run on numpy diff matrices and synthetic activations. The extraction /
ablation orchestration is pod-only and untouched here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")

from refusal_stack.detect.subspace import (  # noqa: E402
    auroc_curve_over_k,
    auroc_from_scores,
    build_refusal_subspace,
    orient_basis,
    per_position_diff_matrix,
    subspace_projection_score,
)


def _diff_matrix(m=6, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, d)).astype(np.float32)


def test_basis_is_orthonormal_pca():
    diff = _diff_matrix()
    basis = build_refusal_subspace(diff, k_max=4, method="pca")
    assert basis.shape == (4, 16)
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(4), atol=1e-4)


def test_basis_is_orthonormal_topk_diff():
    diff = _diff_matrix()
    basis = build_refusal_subspace(diff, k_max=3, method="topk_diff")
    assert basis.shape == (3, 16)
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(3), atol=1e-4)


def test_k_clamped_to_rank():
    # k_max larger than min(m, d) is clamped.
    diff = _diff_matrix(m=3, d=16)
    basis = build_refusal_subspace(diff, k_max=8, method="pca")
    assert basis.shape[0] <= 3


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        build_refusal_subspace(_diff_matrix(), k_max=2, method="nope")


def test_k1_reduces_to_single_direction_projection():
    # For k=1 the subspace score is a plain 1-D projection onto the oriented axis.
    diff = _diff_matrix(m=5, d=8)
    diff_mean = diff.mean(0)
    basis = orient_basis(build_refusal_subspace(diff, k_max=1, method="pca"), diff_mean)
    acts = np.random.default_rng(1).standard_normal((10, 8))
    score = subspace_projection_score(acts, basis)
    expected = acts @ basis[0]
    assert np.allclose(score, expected, atol=1e-5)


def test_orient_basis_points_along_diff_mean():
    diff = _diff_matrix(m=5, d=8)
    diff_mean = diff.mean(0)
    basis = build_refusal_subspace(diff, k_max=4, method="pca")
    oriented = orient_basis(basis, diff_mean)
    # Every oriented axis has non-negative dot with the diff mean.
    assert np.all(oriented @ diff_mean >= -1e-6)


def test_auroc_from_scores_perfect_separation():
    # base (clean) projects high, test (tampered) low -> detector AUROC ~ 1.
    base = np.full(20, 5.0)
    test = np.full(20, -5.0)
    assert auroc_from_scores(base, test) > 0.99


def test_auroc_from_scores_chance():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(200)
    test = rng.standard_normal(200)
    assert 0.4 <= auroc_from_scores(base, test) <= 0.6


def test_auroc_curve_over_k_shape_and_range():
    d = 12
    diff = _diff_matrix(m=6, d=d)
    diff_mean = diff.mean(0)
    basis = build_refusal_subspace(diff, k_max=5, method="pca")
    rng = np.random.default_rng(4)
    # Make base separable from test along the diff mean so AUROC is meaningful.
    base = rng.standard_normal((30, d)) + 3.0 * diff_mean / np.linalg.norm(diff_mean)
    test = rng.standard_normal((30, d)) - 3.0 * diff_mean / np.linalg.norm(diff_mean)
    curve = auroc_curve_over_k(base, test, basis, diff_mean)
    assert [row["k"] for row in curve] == [1, 2, 3, 4, 5]
    for row in curve:
        assert 0.0 <= row["auroc"] <= 1.0
    # k=1 should already separate strongly given the constructed shift.
    assert curve[0]["auroc"] > 0.8


def test_per_position_diff_matrix_2d_and_3d():
    # 2-D input -> single diff row.
    harm2 = np.ones((5, 8))
    harmless2 = np.zeros((5, 8))
    m2 = per_position_diff_matrix(harm2, harmless2)
    assert m2.shape == (1, 8)
    assert np.allclose(m2, 1.0)

    # 3-D input -> one diff row per position.
    harm3 = np.ones((5, 4, 8))
    harmless3 = np.zeros((5, 4, 8))
    m3 = per_position_diff_matrix(harm3, harmless3)
    assert m3.shape == (4, 8)
