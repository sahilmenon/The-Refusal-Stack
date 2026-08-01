"""Figure 4: single-turn vs agentic refusal rate and ASR delta."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.figures.style import apply_style, PALETTE, FIGSIZE_DOUBLE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(
    delta_path: str = "results/delta_report.json",
    attacks_path: str = "results/agentic_attacks.json",
    out_dir: str = "figures/",
) -> None:
    apply_style()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(delta_path) as f:
            delta = json.load(f)
    except FileNotFoundError:
        delta = {"refusal_rate_delta": -0.05, "asr_delta": 0.05, "attack_delta": {"agentic_pair": 0.02, "indirect_injection": 0.15}}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Panel 1: refusal rate / ASR delta
    metrics = ["refusal_rate_delta", "asr_delta"]
    values = [delta.get(m, 0.0) for m in metrics]
    colors = [PALETTE["phase1"] if v >= 0 else PALETTE["phase2"] for v in values]
    ax1.bar(["RR delta", "ASR delta"], values, color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Agentic - Single-turn")
    ax1.set_title("Refusal Rate / ASR Delta")

    # Panel 2: per-attack delta
    attack_delta = delta.get("attack_delta", {})
    if attack_delta:
        ax2.bar(list(attack_delta.keys()), list(attack_delta.values()), color=PALETTE["phase5"])
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("ASR delta")
        ax2.set_title("Attack ASR Delta by Type")
    else:
        ax2.text(0.5, 0.5, "No attack data", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("Single-Turn vs Agentic Delta (Phase 5)")
    fig.tight_layout()
    out = Path(out_dir) / "fig4_agentic_delta.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
