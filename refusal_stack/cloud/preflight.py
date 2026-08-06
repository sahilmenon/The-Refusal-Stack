"""Preflight CLI: verify gated licenses + report the current budget ledger.

Run before the first paid pod:  python -m refusal_stack.cloud.preflight
Exits non-zero if any gated license is still pending, so CI / the /loop can
gate a paid launch on it.
"""

from __future__ import annotations

import argparse
import logging
import sys

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.licenses import all_authorized, check_gated_licenses


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud preflight: licenses + budget")
    parser.add_argument("--token", default=None, help="HF token (defaults to $HF_TOKEN)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("preflight")

    from refusal_stack.cloud.env import load_dotenv

    load_dotenv()

    statuses = check_gated_licenses(token=args.token)
    tracker = CostTracker()
    log.info("Budget: $%.2f spent of $%.2f cap.", tracker.total_usd(), tracker.hard_cap_usd)

    if all_authorized(statuses):
        log.info("Preflight PASS - gated licenses approved, safe to launch a paid pod.")
        sys.exit(0)
    log.error("Preflight FAIL - accept the pending Meta licenses before a paid pod.")
    sys.exit(1)


if __name__ == "__main__":
    main()
