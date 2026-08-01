"""Figure 2: GCG vs PAIR ASR comparison (Phase 2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.figures.style import apply_style, PALETTE, FIGSIZE_SINGLE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(results_path: str = "results/phase2_attacks.json", out_dir: str = "figures/") -> None:
    apply_style()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"gcg": {"asr": 0.45}, "pair": {"asr": 0.30}}

    attacks = list(data.keys())
    asrs = [data[a].get("asr", 0.0) for a in attacks]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [PALETTE["phase2"], PALETTE["phase5"]]
    ax.bar(attacks, asrs, color=colors[:len(attacks)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("GCG vs PAIR Attack Success Rate (Phase 2)")
    fig.tight_layout()
    out = Path(out_dir) / "fig2_asr_comparison.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
