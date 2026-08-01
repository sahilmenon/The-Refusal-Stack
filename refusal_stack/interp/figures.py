from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_layer_separation(scores: dict[int, float], best_layer: int, save_path: str) -> None:
    layers = sorted(scores)
    vals = [scores[l] for l in layers]
    colors = ["crimson" if l == best_layer else "steelblue" for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, vals, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Layer Separation Score (Diff-of-Means)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_sim_heatmap(probe_results: dict, save_path: str) -> None:
    if not probe_results:
        return
    layers = sorted(probe_results)
    vals = [[probe_results[l].cosine_sim_vs_dom for l in layers]]
    fig, ax = plt.subplots(figsize=(12, 1.5))
    im = ax.imshow(vals, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    ax.set_title("Probe vs Diff-of-Means Cosine Similarity")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_refusal_rate(baseline_rr: float, ablated_rr: float, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Baseline", "Ablated"], [baseline_rr, ablated_rr], color=["steelblue", "crimson"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Effect of Directional Ablation on Refusal")
    for bar, v in zip(bars, [baseline_rr, ablated_rr]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_steering_dose_response(alphas, false_refusal_rates, baseline_frr: float, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, false_refusal_rates, marker="o", color="darkorange", label="Steered FRR")
    ax.axhline(baseline_frr, linestyle="--", color="gray", label="Baseline FRR")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel("False-Refusal Rate")
    ax.set_title("Steering Dose-Response")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_probe_accuracy_per_layer(probe_results: dict, best_layer: int, save_path: str) -> None:
    if not probe_results:
        return
    layers = sorted(probe_results)
    accs = [probe_results[l].accuracy for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, accs, marker="o", color="steelblue")
    ax.axvline(best_layer, linestyle="--", color="crimson", label=f"Best layer {best_layer}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title("Linear Probe Accuracy per Layer")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
