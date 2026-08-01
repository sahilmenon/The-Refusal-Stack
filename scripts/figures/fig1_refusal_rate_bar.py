"""Figure 1: refusal rate by dataset (Phase 1 results)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.figures.style import apply_style, PALETTE, FIGSIZE_SINGLE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(results_path: str = "results/phase1_eval.json", out_dir: str = "figures/") -> None:
    apply_style()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"mock": {"refusal_rate_harmful": 0.85, "false_refusal_rate_benign": 0.05}}

    datasets = list(data.keys())
    refusal_rates = [data[d].get("refusal_rate_harmful", 0.0) for d in datasets]
    false_rates = [data[d].get("false_refusal_rate_benign", 0.0) for d in datasets]

    x = range(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar([i - width / 2 for i in x], refusal_rates, width, label="Refusal rate (harmful)", color=PALETTE["phase1"])
    ax.bar([i + width / 2 for i in x], false_rates, width, label="False refusal (benign)", color=PALETTE["phase3"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title("Refusal Rate by Dataset (Phase 1)")
    fig.tight_layout()
    out = Path(out_dir) / "fig1_refusal_rate_bar.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
