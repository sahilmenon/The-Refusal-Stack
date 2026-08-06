from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from refusal_stack.attacks.metrics import queries_to_success_cdf


def make_asr_bar_chart(
    baseline_asr: float, gcg_asr: float, pair_asr: float, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Baseline", "GCG", "PAIR"],
        [baseline_asr, gcg_asr, pair_asr],
        color=["steelblue", "crimson", "darkorange"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate by Method")
    for bar, v in zip(bars, [baseline_asr, gcg_asr, pair_asr]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_queries_to_success_cdf(gcg_results, pair_results, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    gx, gy = queries_to_success_cdf(gcg_results)
    px, py = queries_to_success_cdf(pair_results)
    if len(gx):
        ax.plot(gx, gy, label="GCG", color="crimson")
    if len(px):
        ax.plot(px, py, label="PAIR", color="darkorange")
    ax.set_xlabel("Queries")
    ax.set_ylabel("Fraction Successful")
    ax.legend()
    ax.set_title("Queries to Success CDF")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_attack_success_over_iterations(gcg_step_logs: list[dict], out_path: Path) -> None:
    if not gcg_step_logs:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        fig.savefig(str(out_path))
        plt.close(fig)
        return
    steps = [rec["step"] for rec in gcg_step_logs]
    cumulative_asr = []
    n_success = 0
    for i, rec in enumerate(gcg_step_logs):
        if rec.get("success", False):
            n_success += 1
        cumulative_asr.append(n_success / (i + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, cumulative_asr, color="crimson")
    ax.set_xlabel("Step")
    ax.set_ylabel("Running ASR")
    ax.set_title("Attack Success over Iterations")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_headroom_figure(analysis_dict: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(
        ["Baseline", "GCG", "PAIR"],
        [
            1 - analysis_dict.get("gcg_asr", 0),
            analysis_dict.get("gcg_asr", 0),
            analysis_dict.get("pair_asr", 0),
        ],
        color=["steelblue", "crimson", "darkorange"],
    )
    ax1.set_ylim(0, 1)
    ax1.set_title("ASR by Method")
    metrics = ["Headroom", "Transfer ASR"]
    values = [analysis_dict.get("headroom", 0), analysis_dict.get("transfer_asr", 0)]
    ax2.barh(metrics, values, color=["purple", "teal"])
    ax2.set_xlim(0, 1)
    ax2.set_title("Headroom & Transfer")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
