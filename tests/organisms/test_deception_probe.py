"""CPU-only tests for the deception probe (sklearn on synthetic activations).

Guarded with pytest.importorskip so the suite skips cleanly where numpy/sklearn
are absent. No model, no GPU: the probe seam (fit_deception_probe / probe_auroc)
is fed synthetic activation matrices.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")


def _synthetic_acts(n: int, d: int, shift: float, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) + shift


def test_probe_separates_shifted_classes():
    """Honest and deceptive activations that differ by a mean shift are separable."""
    from refusal_stack.generalize.organisms.deception_probe import fit_deception_probe, probe_auroc

    honest = _synthetic_acts(40, 16, shift=0.0, seed=1)
    deceptive = _synthetic_acts(40, 16, shift=2.5, seed=2)
    clf, cv_acc = fit_deception_probe(honest, deceptive)
    auroc = probe_auroc(clf, honest, deceptive)
    assert auroc > 0.9
    assert 0.0 <= cv_acc <= 1.0


def test_probe_chance_on_identical_distributions():
    """Overlapping classes -> AUROC near chance (probe doesn't hallucinate signal)."""
    from refusal_stack.generalize.organisms.deception_probe import fit_deception_probe, probe_auroc

    honest = _synthetic_acts(60, 16, shift=0.0, seed=3)
    deceptive = _synthetic_acts(60, 16, shift=0.0, seed=4)
    clf, _ = fit_deception_probe(honest, deceptive)
    auroc = probe_auroc(clf, honest, deceptive)
    # In-sample AUROC on pure noise stays modest; certainly not a strong detector.
    assert auroc < 0.85


def test_probe_labels_positive_class_is_deceptive():
    """Class 1 must be the deceptive (sandbagger) set, class 0 the honest control."""
    from refusal_stack.generalize.organisms.deception_probe import fit_deception_probe

    honest = _synthetic_acts(30, 8, shift=-3.0, seed=5)
    deceptive = _synthetic_acts(30, 8, shift=3.0, seed=6)
    clf, _ = fit_deception_probe(honest, deceptive)
    # A clearly-deceptive point should score >0.5 on the positive class.
    prob = clf.predict_proba(np.full((1, 8), 3.0))[0, 1]
    assert prob > 0.5
