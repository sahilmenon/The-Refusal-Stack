"""Cumulative GPU cost tracking with a hard budget cap.

Tracks pod-seconds x hourly rate across phases and enforces the AUD-50 ceiling:
soft alert at US$22, hard cap at US$32. The tracker halts-and-asks before any
launch that would cross the cap, and exposes a self-kill signal a backstop can
poll even when the /loop is dormant.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SOFT_ALERT_USD = 22.0
HARD_CAP_USD = 32.0

# Community-cloud reference rates (USD/hr); escalate to A100 only on OOM.
GPU_HOURLY_USD = {
    "RTX4090": 0.34,
    "A40": 0.40,
    "A100": 1.19,
}


@dataclass
class PhaseCost:
    phase: str
    gpu: str
    seconds: float
    usd: float


@dataclass
class CostTracker:
    """Accumulates per-phase GPU cost and guards the hard cap.

    Persists to ``ledger_path`` so cost survives across separate /loop wake-ups.
    """

    ledger_path: str = "outputs/cost_ledger.json"
    soft_alert_usd: float = SOFT_ALERT_USD
    hard_cap_usd: float = HARD_CAP_USD
    entries: list[PhaseCost] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._load()

    # --- persistence ---------------------------------------------------------
    def _load(self) -> None:
        p = Path(self.ledger_path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.entries = [PhaseCost(**e) for e in data.get("entries", [])]
            except (json.JSONDecodeError, TypeError):
                self.entries = []

    def _save(self) -> None:
        p = Path(self.ledger_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"entries": [vars(e) for e in self.entries]}, indent=2))

    # --- accounting ----------------------------------------------------------
    @staticmethod
    def estimate_usd(gpu: str, seconds: float) -> float:
        rate = GPU_HOURLY_USD.get(gpu, GPU_HOURLY_USD["A40"])
        return rate * (seconds / 3600.0)

    def total_usd(self) -> float:
        return sum(e.usd for e in self.entries)

    def record(self, phase: str, gpu: str, seconds: float) -> PhaseCost:
        usd = self.estimate_usd(gpu, seconds)
        entry = PhaseCost(phase=phase, gpu=gpu, seconds=seconds, usd=usd)
        self.entries.append(entry)
        self._save()
        total = self.total_usd()
        if total >= self.hard_cap_usd:
            logger.error("HARD CAP breached: $%.2f >= $%.2f — kill live pods now.", total, self.hard_cap_usd)
        elif total >= self.soft_alert_usd:
            logger.warning("Soft alert: $%.2f >= $%.2f — approaching the cap.", total, self.soft_alert_usd)
        else:
            logger.info("Phase %s cost $%.2f; cumulative $%.2f / $%.2f.", phase, usd, total, self.hard_cap_usd)
        return entry

    # --- guards --------------------------------------------------------------
    def would_exceed_cap(self, gpu: str, projected_seconds: float) -> bool:
        return self.total_usd() + self.estimate_usd(gpu, projected_seconds) > self.hard_cap_usd

    def check_before_launch(self, gpu: str, projected_seconds: float) -> bool:
        """Return True if a launch is within budget; False means halt-and-ask.

        Callers must not launch a paid pod when this returns False.
        """
        projected_total = self.total_usd() + self.estimate_usd(gpu, projected_seconds)
        if projected_total > self.hard_cap_usd:
            logger.error(
                "Launch BLOCKED: projected $%.2f would cross the $%.2f cap (current $%.2f).",
                projected_total, self.hard_cap_usd, self.total_usd(),
            )
            return False
        if projected_total > self.soft_alert_usd:
            logger.warning("Launch OK but projected $%.2f crosses the soft alert $%.2f.", projected_total, self.soft_alert_usd)
        return True

    def at_cap(self) -> bool:
        """Backstop signal: a live-pod watchdog polls this to self-kill."""
        return self.total_usd() >= self.hard_cap_usd

    def log_to_wandb(self) -> None:
        try:
            import wandb

            wandb.log({"cost/usd_spent": self.total_usd()})
        except Exception:  # noqa: BLE001 — tracking is best-effort
            pass
