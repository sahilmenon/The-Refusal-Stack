"""CLI: 7C obfuscated-activation adaptive attack (POD-ONLY runtime).

Runs the continuous attack static (no penalty) vs adaptive (detector-projection
penalty) on held-out harmful prompts and writes outputs/obfuscated/result.json.
"""
from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    from refusal_stack.attacks.obfuscated import run_obfuscated
    from refusal_stack.detect.robustness_config import load_robustness_config

    cfg = load_robustness_config(args.config)
    run_obfuscated(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
