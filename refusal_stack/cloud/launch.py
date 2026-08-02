"""One-command pod launch for a phase, with the money checkpoint built in.

Without ``--yes`` this is a DRY RUN: it verifies gated licenses and that the
projected spend stays under the cap, then reports what *would* launch — no pod
is created, nothing is billed. Passing ``--yes`` is the explicit spend consent
that actually provisions the pod.

    python -m refusal_stack.cloud.launch --phase eval --make-target eval        # dry run
    python -m refusal_stack.cloud.launch --phase eval --make-target eval --yes  # spends
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.licenses import gate_paid_pod
from refusal_stack.cloud.runpod import run_phase

log = logging.getLogger("launch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one phase on an ephemeral pod")
    parser.add_argument("--phase", required=True, help="Phase label for the cost ledger")
    parser.add_argument("--make-target", required=True, help="make target to run on the pod")
    parser.add_argument("--gpu", default="RTX4090", help="Cost key: RTX4090 | A40 | A100")
    parser.add_argument("--projected-seconds", type=float, default=1800.0)
    parser.add_argument("--volume", default=None, help="Optional network-volume id")
    parser.add_argument("--yes", action="store_true", help="Consent to spend and actually launch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tracker = CostTracker()
    est = CostTracker.estimate_usd(args.gpu, args.projected_seconds)

    # Gate 1: licenses. Gate 2: budget projection.
    licenses_ok = gate_paid_pod()
    budget_ok = tracker.check_before_launch(args.gpu, args.projected_seconds)

    log.info(
        "Phase '%s' -> make %s on %s: est $%.2f, cumulative would be $%.2f / $%.2f",
        args.phase, args.make_target, args.gpu, est, tracker.total_usd() + est, tracker.hard_cap_usd,
    )

    if not (licenses_ok and budget_ok):
        log.error("Gates FAILED (licenses=%s, budget=%s) — not launching.", licenses_ok, budget_ok)
        sys.exit(1)

    if not args.yes:
        log.info("DRY RUN — gates pass. Re-run with --yes to spend ~$%.2f and launch.", est)
        sys.exit(0)

    log.warning("Launching paid pod (consented via --yes)...")
    summary = run_phase(
        args.phase, args.make_target, gpu=args.gpu,
        projected_seconds=args.projected_seconds, volume=args.volume,
    )
    log.info("Phase complete: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
