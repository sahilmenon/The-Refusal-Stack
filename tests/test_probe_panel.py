"""CPU-only tests for 7B probe-panel pure math.

The mass-mean direction, projection AUROC, length-control, and panel assembly are
numpy-only. The extraction / logistic-fit / SAE / causal-validation paths are
pod-only and untouched here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")

from refusal_stack.detect.probe_panel import (  # noqa: E402
    ProbeEntry,
    assemble_panel,
    length_control_prompts,
    mass_mean_direction,
    projection_auroc,
)


def test_mass_mean_direction_unit_and_points_to_harmful():
    rng = np.random.default_rng(0)
    d = 16
    axis = np.zeros(d)
    axis[0] = 1.0
    harmful = rng.standard_normal((30, d)) + 4.0 * axis
    harmless = rng.standard_normal((30, d)) - 4.0 * axis
    mm = mass_mean_direction(harmful, harmless)
    assert abs(np.linalg.norm(mm) - 1.0) < 1e-5
    # Direction should align with the +axis (harmful side).
    assert mm[0] > 0.9


def test_projection_auroc_perfect():
    d = 8
    axis = np.zeros(d)
    axis[0] = 1.0
    harmful = np.tile(3.0 * axis, (20, 1))
    harmless = np.tile(-3.0 * axis, (20, 1))
    assert projection_auroc(axis, harmful, harmless) > 0.99


def test_projection_auroc_sign_invariant():
    # A flipped direction must not report < 0.5 (we orient to the better side).
    d = 8
    axis = np.zeros(d)
    axis[0] = 1.0
    harmful = np.tile(3.0 * axis, (20, 1))
    harmless = np.tile(-3.0 * axis, (20, 1))
    assert projection_auroc(-axis, harmful, harmless) > 0.99


def test_projection_auroc_chance():
    rng = np.random.default_rng(2)
    d = 16
    harmful = rng.standard_normal((100, d))
    harmless = rng.standard_normal((100, d))
    direction = rng.standard_normal(d)
    assert 0.4 <= projection_auroc(direction, harmful, harmless) <= 0.6


def test_length_control_pads_and_trims():
    prompts = ["a b c", " ".join(["w"] * 40)]
    out = length_control_prompts(prompts, target_words=24)
    assert all(len(p.split()) == 24 for p in out)


def test_assemble_panel_best_and_lift():
    entries = [
        ProbeEntry("unsupervised", 0.70),
        ProbeEntry("mass_mean", 0.75),
        ProbeEntry("logistic", 0.88),
    ]
    panel = assemble_panel(entries)
    assert panel["best_probe"] == "logistic"
    assert panel["best_auroc"] == 0.88
    assert panel["unsupervised_auroc"] == 0.70
    assert abs(panel["supervised_lift_over_unsupervised"] - 0.18) < 1e-9
    assert len(panel["probes"]) == 3


def test_probe_entry_to_dict_carries_causal_fields():
    e = ProbeEntry(
        "logistic",
        0.9,
        causal_refusal_drop=0.5,
        causal_valid=True,
        paraphrase_auroc=0.85,
        invariance_gap=0.05,
    )
    d = e.to_dict()
    assert d["causal_valid"] is True
    assert d["causal_refusal_drop"] == 0.5
    assert d["invariance_gap"] == 0.05
