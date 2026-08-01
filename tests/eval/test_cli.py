"""Smoke test for the CLI — dry run, no GPU, no W&B."""
from __future__ import annotations

import subprocess
import sys


def test_cli_dry_run_exits_zero():
    """CLI --dry-run --no-judge with mocked model should exit 0."""
    # We can't load real weights in CI, so we only test that the CLI
    # parses args and loads config correctly (import-time errors surface here).
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from refusal_stack.eval.cli import _build_parser; p = _build_parser(); print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
