"""Unit tests for the GPU cost tracker."""

from __future__ import annotations

from refusal_stack.cloud.cost import HARD_CAP_USD, CostTracker


def _tracker(tmp_path):
    return CostTracker(ledger_path=str(tmp_path / "ledger.json"))


def test_estimate_usd_matches_rate():
    # A40 at $0.40/hr for one hour.
    assert abs(CostTracker.estimate_usd("A40", 3600.0) - 0.40) < 1e-9


def test_record_accumulates_and_persists(tmp_path):
    t = _tracker(tmp_path)
    t.record("eval", "A40", 1800.0)  # $0.20
    t.record("attack", "A40", 1800.0)  # $0.20
    assert abs(t.total_usd() - 0.40) < 1e-9
    # A fresh tracker on the same ledger reloads the running total.
    reloaded = CostTracker(ledger_path=t.ledger_path)
    assert abs(reloaded.total_usd() - 0.40) < 1e-9


def test_check_before_launch_blocks_over_cap(tmp_path):
    t = _tracker(tmp_path)
    # ~29 hours on A40 ~= $11.6; project a huge run that would cross the cap.
    assert t.check_before_launch("A40", 3600.0) is True
    huge_seconds = (HARD_CAP_USD / 0.40) * 3600.0 + 3600.0
    assert t.check_before_launch("A40", huge_seconds) is False


def test_at_cap_backstop(tmp_path):
    t = _tracker(tmp_path)
    t.record("gcg", "A100", (HARD_CAP_USD / 1.19) * 3600.0 + 3600.0)
    assert t.at_cap() is True

    fresh = CostTracker(ledger_path=str(tmp_path / "empty.json"))
    assert fresh.at_cap() is False
