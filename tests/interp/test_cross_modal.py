"""CPU-only tests for 7E cross-modal refusal-direction transfer.

No model / GPU: exercises the pure direction + overlap math on numpy arrays. The
Chameleon generation/ablation transfer path in run_cross_modal is pod-only and
not tested here.
"""

import numpy as np
import pytest

pytest.importorskip("numpy")

from refusal_stack.interp.vlm.cross_modal import (
    direction_from_contrast,
    principal_angle_degrees,
    subspace_overlap,
    transfer_summary,
)


def test_direction_from_contrast_is_unit_and_points_right_way():
    rng = np.random.default_rng(0)
    d = 32
    axis = np.zeros(d, dtype=np.float32)
    axis[3] = 1.0
    harmful = rng.standard_normal((20, d)).astype(np.float32) + 5.0 * axis
    harmless = rng.standard_normal((20, d)).astype(np.float32)
    r = direction_from_contrast(harmful, harmless)
    assert abs(np.linalg.norm(r) - 1.0) < 1e-5
    # dominant component is the injected axis
    assert np.argmax(np.abs(r)) == 3


def test_subspace_overlap_collinear_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-2.0, 0.0, 0.0])  # anti-parallel -> same line
    c = np.array([0.0, 1.0, 0.0])
    assert subspace_overlap(a, b) == pytest.approx(1.0)
    assert subspace_overlap(a, c) == pytest.approx(0.0)


def test_principal_angle_degrees():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert principal_angle_degrees(a, b) == pytest.approx(90.0)
    # collinear -> ~0 degrees (allow tiny arccos rounding near 1.0)
    assert principal_angle_degrees(a, np.array([2.0, 0.0])) == pytest.approx(0.0, abs=0.1)


def test_transfer_summary_keys():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.9, 0.1, 0.0])
    s = transfer_summary(a, b)
    assert set(s) == {"cosine_text_image", "subspace_overlap", "principal_angle_deg"}
    assert 0.0 <= s["subspace_overlap"] <= 1.0
