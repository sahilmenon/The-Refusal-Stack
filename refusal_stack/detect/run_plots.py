"""CLI: generate all Phase 4 figures."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/")
    args = parser.parse_args()

    from refusal_stack.detect.evasion import run_evasion_sweep
    from refusal_stack.detect.plots import (
        plot_evasion_sweep,
        plot_projection_histograms,
        plot_refusal_rates,
        plot_roc_curve,
        plot_tpr_fpr_table_heatmap,
    )
    from refusal_stack.detect.scorer import TamperDetector

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load projection arrays (best effort — skip if missing)
    proj_dir = Path("outputs/projections")
    base_path = proj_dir / "base_projections.npy"
    mal_path = proj_dir / "malicious_projections.npy"
    ben_path = proj_dir / "benign_control_projections.npy"

    if base_path.exists() and mal_path.exists() and ben_path.exists():
        base = np.load(str(base_path))
        mal = np.load(str(mal_path))
        ben = np.load(str(ben_path))

        detector = TamperDetector(base, mal)
        auroc = detector.compute_auroc()
        detector.fit_threshold(0.05)

        plot_projection_histograms(base, mal, ben, str(out / "phase4_projection_histograms.png"), detector.threshold)
        plot_roc_curve(base, mal, 0.05, auroc, str(out / "phase4_roc_curve.png"))

        evasion_df = run_evasion_sweep(base, mal)
        plot_evasion_sweep(evasion_df, str(out / "phase4_evasion_sweep.png"))

        thresholds = [detector.threshold * x for x in [0.5, 0.75, 1.0, 1.25, 1.5]]
        tpr_fpr_df = detector.compute_tpr_fpr_table(thresholds)
        plot_tpr_fpr_table_heatmap(tpr_fpr_df, str(out / "phase4_tpr_fpr_heatmap.png"))
    else:
        log.warning("Projection arrays not found; skipping histogram/ROC/evasion figures")

    # Load eval results
    eval_path = Path("logs/phase4_eval_results.json")
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)
        plot_refusal_rates(eval_results, str(out / "phase4_refusal_rates.png"))
    else:
        log.warning("Phase 4 eval results not found; skipping refusal rate figure")

    log.info(f"Figures written to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
