"""Cheapest end-to-end lifecycle validation (~1 cent).

Creates a pod, runs a trivial command over SSH, and terminates — proving the
create -> ssh -> terminate plumbing works before trusting it with a real phase.
Requires --yes to actually spend; without it, reports what it would do.

    python -m refusal_stack.cloud.selftest          # dry run
    python -m refusal_stack.cloud.selftest --yes    # spends ~1 cent
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheap pod lifecycle self-test")
    parser.add_argument("--gpu", default="RTX4090")
    parser.add_argument("--yes", action="store_true", help="Consent to spend ~1 cent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("selftest")

    import os

    from refusal_stack.cloud.env import load_dotenv

    load_dotenv()
    if not os.environ.get("RUNPOD_API_KEY"):
        log.error("RUNPOD_API_KEY not set (checked env + .env) — cannot reach RunPod.")
        sys.exit(1)

    if not args.yes:
        log.info("DRY RUN — would create a %s pod, ssh `echo`, and terminate (~1 cent). "
                 "Re-run with --yes to spend.", args.gpu)
        sys.exit(0)

    from refusal_stack.cloud.runpod import self_test

    log.warning("Launching self-test pod (consented via --yes)...")
    result = self_test(gpu=args.gpu)
    log.info("Self-test result: %s", json.dumps(result, default=str))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
