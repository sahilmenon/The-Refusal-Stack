"""Fetch W&B run summaries and write results/wandb_summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping")
        return

    api = wandb.Api()
    runs = api.runs("the-refusal-stack")
    summary: dict = {}

    for run in runs:
        tags = run.tags or []
        phase = next((t for t in tags if t.startswith("phase")), run.name or "unknown")
        summary[run.id] = {
            "name": run.name,
            "phase": phase,
            "state": run.state,
            "summary": dict(run.summary),
        }

    Path("results").mkdir(exist_ok=True)
    with open("results/wandb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Fetched {len(summary)} runs")
    print("| Run | Phase | State |")
    print("|-----|-------|-------|")
    for run_id, info in summary.items():
        print(f"| {info['name']} | {info['phase']} | {info['state']} |")


if __name__ == "__main__":
    main()
