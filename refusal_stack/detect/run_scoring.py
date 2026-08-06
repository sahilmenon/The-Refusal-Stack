"""CLI: score base vs test projections."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-projections", required=True)
    parser.add_argument("--test-projections", required=True)
    parser.add_argument("--test-label", required=True)
    parser.add_argument("--detect-config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.scorer import TamperDetector

    cfg = load_detect_config(args.detect_config)
    base = np.load(args.base_projections)
    test = np.load(args.test_projections)

    detector = TamperDetector(base, test)
    auroc = detector.compute_auroc()
    separation = detector.compute_separation()
    threshold = detector.fit_threshold(cfg.threshold_fpr_target)

    classified_base = detector.classify(base)
    fpr_base = float(np.mean(classified_base))

    classified_test = detector.classify(test)
    tpr = float(np.mean(classified_test))

    results = {
        "label": args.test_label,
        "auroc": auroc,
        "threshold": threshold,
        "tpr_at_target_fpr": tpr,
        "base_fpr": fpr_base,
        **separation,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_{args.test_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"AUROC={auroc:.3f}  threshold={threshold:.4f}  TPR={tpr:.3f}")
    log.info(f"Results saved to {out_path}")

    # Merge into the canonical, git-tracked results/phase4_detect.json keyed by
    # test label, so each detector run (malicious, benign_control) accumulates
    # into the one file downstream figures/repro read.
    canonical = Path("results/phase4_detect.json")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    merged: dict = {}
    if canonical.exists():
        try:
            merged = json.loads(canonical.read_text())
        except json.JSONDecodeError:
            merged = {}
    merged[args.test_label] = results
    with open(canonical, "w") as f:
        json.dump(merged, f, indent=2)
    log.info(f"Canonical detector results merged into {canonical}")

    try:
        import wandb

        wandb.log(
            {
                f"detect/{args.test_label}_auroc": auroc,
                f"detect/{args.test_label}_cohen_d": separation["cohen_d"],
                "detect/threshold": threshold,
                f"detect/{args.test_label}_tpr_at_target_fpr": tpr,
            }
        )
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
