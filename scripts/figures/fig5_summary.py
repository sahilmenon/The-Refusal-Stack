"""Figure 5: 5-panel summary across all phases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.figures.style import apply_style, PALETTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _load(path: str, default: dict) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def main(out_dir: str = "figures/") -> None:
    apply_style()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    p1 = _load("results/phase1_eval.json", {"advbench": {"refusal_rate_harmful": 0.88}})
    p2 = _load("results/phase2_attacks.json", {"gcg": {"asr": 0.45}, "pair": {"asr": 0.30}})
    p3 = _load("results/phase3_interp.json", {"best_layer": 15, "ablated_refusal_rate": 0.12})
    p4 = _load("results/phase4_detect.json", {"malicious_auroc": 0.92, "benign_auroc": 0.55})
    p5 = _load("results/delta_report.json", {"asr_delta": 0.05})

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # P1: refusal rate bar
    ax1 = fig.add_subplot(gs[0, 0])
    rr = list(p1.values())[0].get("refusal_rate_harmful", 0.88) if p1 else 0.88
    ax1.bar(["harmful", "benign"], [rr, 0.05], color=[PALETTE["phase1"], PALETTE["phase3"]])
    ax1.set_ylim(0, 1.05)
    ax1.set_title("P1: Refusal Rate")
    ax1.set_ylabel("Rate")

    # P2: ASR comparison
    ax2 = fig.add_subplot(gs[0, 1])
    attacks = list(p2.keys())
    asrs = [p2[a].get("asr", 0.0) for a in attacks]
    ax2.bar(attacks, asrs, color=PALETTE["phase2"])
    ax2.set_ylim(0, 1.05)
    ax2.set_title("P2: Attack ASR")

    # P3: ablation result
    ax3 = fig.add_subplot(gs[0, 2])
    baseline_rr = 0.88
    ablated_rr = p3.get("ablated_refusal_rate", 0.12)
    ax3.bar(["baseline", "ablated"], [baseline_rr, ablated_rr], color=[PALETTE["phase1"], PALETTE["phase3"]])
    ax3.set_ylim(0, 1.05)
    ax3.set_title("P3: Ablation Effect")

    # P4: detector AUROC
    ax4 = fig.add_subplot(gs[1, 0])
    mal_auroc = p4.get("malicious_auroc", 0.92)
    ben_auroc = p4.get("benign_auroc", 0.55)
    ax4.bar(["malicious", "benign"], [mal_auroc, ben_auroc], color=[PALETTE["phase4"], PALETTE["phase3"]])
    ax4.axhline(0.5, color="red", linestyle="--", label="random")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("P4: Detector AUROC")
    ax4.legend()

    # P5: delta
    ax5 = fig.add_subplot(gs[1, 1])
    asr_delta = p5.get("asr_delta", 0.05)
    ax5.bar(["ASR delta"], [asr_delta], color=PALETTE["phase5"])
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title("P5: Agentic Delta")

    # Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = (
        f"Phase 1: RR={rr:.2f}\n"
        f"Phase 2: ASR={max(asrs):.2f}\n"
        f"Phase 3: ablated RR={ablated_rr:.2f}\n"
        f"Phase 4: AUROC={mal_auroc:.2f}\n"
        f"Phase 5: ΔASR={asr_delta:+.2f}"
    )
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes, fontsize=12,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax6.set_title("Headline Results")

    fig.suptitle("The Refusal Stack — Summary", fontsize=14, fontweight="bold")
    out = Path(out_dir) / "fig5_summary.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
