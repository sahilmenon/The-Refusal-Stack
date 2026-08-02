"""Reproducibility checker: re-run a phase's make target and compare metrics."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_PHASE_TARGETS: dict[int, str] = {
    1: "eval",
    2: "attack",
    3: "interp",
    4: "finetune detect",
    5: "eval-agentic attack-agentic analyze-delta",
}

_PHASE_OUTPUT: dict[int, str] = {
    1: "results/phase1_eval.json",
    2: "results/phase2_attacks.json",
    3: "results/interp_summary.json",
    4: "logs/phase4_eval_results.json",
    5: "results/delta_report.json",
}


def _load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _metrics_match(actual: dict, expected: dict, atol: float = 0.01) -> bool:
    for key, exp_val in expected.items():
        if not isinstance(exp_val, (int, float)):
            continue
        act_val = actual.get(key)
        if act_val is None:
            log.warning("Key %s missing from actual results", key)
            return False
        if abs(act_val - exp_val) > atol:
            log.warning("Key %s: actual=%s expected=%s (atol=%s)", key, act_val, exp_val, atol)
            return False
    return True


def check_repro(phase: int, atol: float = 0.01) -> bool:
    """Re-run phase's make target(s), compare output against expected JSON.

    Returns True if all headline metrics match within atol.
    """
    targets = _PHASE_TARGETS.get(phase)
    if targets is None:
        raise ValueError(f"No make target registered for phase {phase}")

    expected_path = Path(f"results/expected/{phase}_expected.json")
    if not expected_path.exists():
        log.error("Expected file not found: %s", expected_path)
        return False

    expected = _load_json(expected_path)

    for target in targets.split():
        log.info("Running: make %s", target)
        proc = subprocess.run(
            ["make", target],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            log.error("make %s failed:\n%s", target, proc.stderr)
            return False

    output_path = Path(_PHASE_OUTPUT[phase])
    if not output_path.exists():
        log.error("Output file not found after make run: %s", output_path)
        return False

    actual = _load_json(output_path)
    passed = _metrics_match(actual, expected, atol=atol)
    if passed:
        log.info("Phase %d repro check PASSED (atol=%.3f)", phase, atol)
    else:
        log.error("Phase %d repro check FAILED", phase)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducibility checker")
    parser.add_argument("--phase", type=int, required=True, help="Phase number (1–5)")
    parser.add_argument("--atol", type=float, default=0.01, help="Absolute tolerance for metric comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    passed = check_repro(args.phase, atol=args.atol)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
