"""Shared matplotlib style for all Phase 5 figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "phase1": "#4878D0",
    "phase2": "#EE854A",
    "phase3": "#6ACC65",
    "phase4": "#D65F5F",
    "phase5": "#956CB4",
    "base": "#4878D0",
    "malicious": "#EE854A",
    "benign_control": "#6ACC65",
}

FIGSIZE_SINGLE = (8, 5)
FIGSIZE_DOUBLE = (12, 5)


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
    })
