"""Print and save DeltaReport."""
from __future__ import annotations

import json
from pathlib import Path

from refusal_stack.agent.results import DeltaReport


def print_delta_table(report: DeltaReport) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Single-Turn vs Agentic Delta")
        table.add_column("Metric")
        table.add_column("Delta", justify="right")
        table.add_row("refusal_rate_delta", f"{report.refusal_rate_delta:+.4f}")
        table.add_row("asr_delta", f"{report.asr_delta:+.4f}")
        for k, v in report.attack_delta.items():
            table.add_row(f"attack_{k}_delta", f"{v:+.4f}")
        Console().print(table)
    except ImportError:
        print(f"refusal_rate_delta: {report.refusal_rate_delta:+.4f}")
        print(f"asr_delta: {report.asr_delta:+.4f}")


def save_delta_json(report: DeltaReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
