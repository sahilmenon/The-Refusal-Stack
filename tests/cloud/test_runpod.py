"""Unit tests for the pod lifecycle: teardown guarantee + cost/cap guards."""
from __future__ import annotations

import pytest

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.runpod import PodError, run_phase


class FakeClient:
    def __init__(self, fail_on_exec: bool = False):
        self.fail_on_exec = fail_on_exec
        self.calls: list[str] = []

    def create_pod(self, gpu, volume=None, image=None, env=None, compute_type="GPU", container_disk_gb=40):
        self.calls.append("create")
        return "pod123"

    def wait_for_ssh(self, pod_id, timeout_s=300.0, interval_s=15.0):
        self.calls.append("wait")
        return ("1.2.3.4", 22222)

    def bootstrap(self, pod_id, host, port, extras="."):
        self.calls.append("bootstrap")

    def exec(self, pod_id, command, timeout=None):
        self.calls.append("exec")
        if self.fail_on_exec:
            raise RuntimeError("boom")
        return "done"

    def sync_results(self, pod_id, remote="/workspace/repo", local="."):
        self.calls.append("sync")

    def terminate_pod(self, pod_id):
        self.calls.append("terminate")


def test_run_phase_happy_path(tmp_path):
    client = FakeClient()
    tracker = CostTracker(ledger_path=str(tmp_path / "ledger.json"))
    summary = run_phase(
        "eval", "eval", gpu="A40", projected_seconds=60,
        client=client, tracker=tracker, require_licenses=False,
    )
    assert client.calls == ["create", "wait", "bootstrap", "exec", "sync", "terminate"]
    assert summary["pod_id"] == "pod123"
    assert tracker.total_usd() > 0


def test_terminate_runs_even_when_exec_fails(tmp_path):
    client = FakeClient(fail_on_exec=True)
    tracker = CostTracker(ledger_path=str(tmp_path / "ledger.json"))
    with pytest.raises(RuntimeError, match="boom"):
        run_phase(
            "attack", "attack", gpu="A40", projected_seconds=60,
            client=client, tracker=tracker, require_licenses=False,
        )
    # The finally block must still tear the pod down (no runaway billing).
    assert "terminate" in client.calls
    # And the cost of the (failed) run is still recorded.
    assert tracker.total_usd() > 0


def test_launch_blocked_when_projection_exceeds_cap(tmp_path):
    client = FakeClient()
    tracker = CostTracker(ledger_path=str(tmp_path / "ledger.json"))
    huge_seconds = (tracker.hard_cap_usd / 0.40) * 3600.0 + 3600.0
    with pytest.raises(PodError, match="hard cap"):
        run_phase(
            "gcg", "attack", gpu="A40", projected_seconds=huge_seconds,
            client=client, tracker=tracker, require_licenses=False,
        )
    # No pod should have been created.
    assert client.calls == []
