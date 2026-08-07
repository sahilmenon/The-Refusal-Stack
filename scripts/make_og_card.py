"""Render the dashboard social-share card (dashboard/og.png, 1200x630).

Matches the site's forensic dark theme so a shared link unfurls on-brand.
Run: python scripts/make_og_card.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BG = "#0b0e13"
INK = "#dde5ee"
TEAL = "#4fd1c5"
DIM = "#808d9e"
MUTED = "#9aa7b5"
MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"


def main(out: str = "dashboard/og.png") -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")
    ax.set_facecolor(BG)

    # faint teal glow in the top-right, echoing the site's hero gradient
    for r, a in [(460, 0.05), (320, 0.05), (190, 0.05)]:
        ax.add_patch(Circle((1180, 690), r, color=TEAL, alpha=a, ec="none"))

    # brand row
    ax.add_patch(Circle((74, 566), 8, color=TEAL, ec="none"))
    ax.text(95, 566, "FORENSIC", color=INK, fontsize=16, family=MONO, fontweight="bold", va="center")
    ax.text(220, 566, "/ the refusal stack", color=DIM, fontsize=14, family=MONO, va="center")

    # headline
    ax.text(72, 452, "An autopsy of a", color=INK, fontsize=60, family=SANS, fontweight="bold", va="center")
    ax.text(72, 380, "safety behaviour.", color=INK, fontsize=60, family=SANS, fontweight="bold", va="center")

    # what and result
    ax.text(74, 288, "A covert fine-tune strips an open model's safety.", color=MUTED, fontsize=21, family=SANS, va="center")
    ax.text(74, 252, "We catch it from the model's own activations.", color=MUTED, fontsize=21, family=SANS, va="center")

    ax.text(74, 182, "AUROC 0.956", color=TEAL, fontsize=30, family=MONO, fontweight="bold", va="center")
    ax.text(384, 182, "activation-space tamper detection", color=DIM, fontsize=17, family=MONO, va="center")

    # footer
    ax.text(74, 66, "Sahil Menon", color=INK, fontsize=17, family=SANS, fontweight="bold", va="center")
    ax.text(270, 66, "AI-safety research  ·  forensic.sahilmenon.com", color=DIM, fontsize=15, family=SANS, va="center")

    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
