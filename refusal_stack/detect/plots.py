"""Figures for Phase 4 detection results."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_refusal_rates(results: dict, out_path: str) -> None:
    models = list(results.keys())
    rates = [results[m].get("refusal_rate", 0.0) for m in models]
    asrs = [results[m].get("asr", 0.0) for m in models]
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], rates, width, label="refusal_rate", color="#4878D0")
    ax.bar([i + width / 2 for i in x], asrs, width, label="asr", color="#EE854A")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title("Refusal Rate and ASR by Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_projection_histograms(
    base_proj: np.ndarray,
    malicious_proj: np.ndarray,
    benign_proj: np.ndarray,
    out_path: str,
    threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for data, label, color in [
        (base_proj, "base", "#4878D0"),
        (malicious_proj, "malicious", "#EE854A"),
        (benign_proj, "benign_control", "#6ACC65"),
    ]:
        ax.hist(data, bins=30, alpha=0.5, label=label, color=color, density=True)
    if threshold is not None:
        ax.axvline(-threshold, color="red", linestyle="--", label=f"threshold={-threshold:.3f}")
    ax.set_xlabel("Refusal direction projection")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Refusal Direction Projection by Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    base_proj: np.ndarray,
    malicious_proj: np.ndarray,
    fpr_target: float,
    auroc: float,
    out_path: str,
) -> None:
    from sklearn.metrics import roc_curve

    # Match the detector's convention (scorer.py): base=0 (clean), malicious=1
    # (tampered), scores negated so the tampered class scores higher. The old
    # [1]*base + [0]*malicious inverted this, so the plotted curve was the mirror
    # image (AUROC<0.5) while the title showed the correct number.
    y_true = np.array([0] * len(base_proj) + [1] * len(malicious_proj))
    scores = np.concatenate([-base_proj, -malicious_proj])
    fprs, tprs, _ = roc_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, color="#4878D0", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    idx = int(np.argmin(np.abs(fprs - fpr_target)))
    ax.scatter([fprs[idx]], [tprs[idx]], color="red", zorder=5,
               label=f"FPR={fpr_target}, TPR={tprs[idx]:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUROC={auroc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_evasion_sweep(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["noise_scale"], df["auroc"], marker="o", color="#4878D0")
    ax.axhline(0.5, color="red", linestyle="--", label="random chance")
    ax.set_xlabel("Noise scale")
    ax.set_ylabel("AUROC")
    ax.set_title("Detector AUROC vs Evasion Noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tpr_fpr_table_heatmap(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = df.set_index("threshold")[["tpr", "fpr"]]
    im = ax.imshow(pivot.values.T, aspect="auto", cmap="RdYlGn")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TPR", "FPR"])
    ax.set_xlabel("Threshold index")
    ax.set_title("TPR / FPR across thresholds")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
