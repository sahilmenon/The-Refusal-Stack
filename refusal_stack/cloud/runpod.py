"""Thin RunPod lifecycle wrapper: create -> run one make target -> sync -> kill.

Every GPU phase is ephemeral. ``run_phase`` guarantees ``terminate_pod`` runs
in a ``finally`` block so a crashed run never leaves a pod billing, records the
phase cost, and refuses to launch if the projected spend would cross the cap.

The API layer shells out to ``runpodctl`` when present; the orchestration and
guards are unit-testable without a real account by injecting a fake client.
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.licenses import gate_paid_pod

logger = logging.getLogger(__name__)


class PodError(RuntimeError):
    pass


@dataclass
class Pod:
    pod_id: str
    gpu: str
    created_at: float


class RunPodClient:
    """Shells out to ``runpodctl``. Swap in a fake in tests."""

    def create_pod(self, gpu: str, volume: str | None = None) -> str:
        cmd = ["runpodctl", "create", "pod", "--gpuType", gpu, "--imageName", "refusal-stack:gpu"]
        if volume:
            cmd += ["--volumePath", volume]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pod_id = _parse_pod_id(out.stdout)
        if not pod_id:
            raise PodError(f"Could not parse pod id from: {out.stdout!r}")
        return pod_id

    def exec(self, pod_id: str, command: str) -> str:
        out = subprocess.run(
            ["runpodctl", "exec", "python", "--pod_id", pod_id, "--", "bash", "-lc", command],
            capture_output=True, text=True, check=True,
        )
        return out.stdout

    def sync_results(self, pod_id: str, remote: str = "/workspace/repo", local: str = ".") -> None:
        for sub in ("results", "figures", "artifacts"):
            subprocess.run(
                ["runpodctl", "receive", f"{pod_id}:{remote}/{sub}", f"{local}/{sub}"],
                capture_output=True, text=True, check=False,
            )

    def terminate_pod(self, pod_id: str) -> None:
        subprocess.run(["runpodctl", "remove", "pod", pod_id], capture_output=True, text=True, check=False)


def _parse_pod_id(stdout: str) -> str:
    # runpodctl prints e.g. 'pod "abc123" created'
    import re

    m = re.search(r'"([a-z0-9]+)"', stdout)
    return m.group(1) if m else ""


def run_phase(
    phase: str,
    make_target: str,
    gpu: str = "A40",
    projected_seconds: float = 1800.0,
    volume: str | None = "/workspace",
    client: RunPodClient | None = None,
    tracker: CostTracker | None = None,
    require_licenses: bool = True,
) -> dict:
    """Run one phase on an ephemeral pod, guaranteeing teardown and cost accounting.

    Returns a summary dict. Refuses to launch if licenses are pending or the
    projected spend would cross the hard cap.
    """
    client = client or RunPodClient()
    tracker = tracker or CostTracker()

    if require_licenses and not gate_paid_pod():
        raise PodError("Gated licenses not approved — refusing to launch a paid pod.")

    if not tracker.check_before_launch(gpu, projected_seconds):
        raise PodError("Projected spend would cross the hard cap — halting. Confirm before continuing.")

    pod_id = client.create_pod(gpu, volume=volume)
    started = time.monotonic()
    logger.info("Pod %s up (%s) — running: make %s", pod_id, gpu, make_target)
    status = "ok"
    try:
        client.exec(pod_id, f"cd /workspace/repo && make {make_target}")
        client.sync_results(pod_id)
    except Exception as exc:  # noqa: BLE001 — teardown must still run
        status = f"error: {exc}"
        logger.error("Phase %s failed: %s", phase, exc)
        raise
    finally:
        elapsed = time.monotonic() - started
        client.terminate_pod(pod_id)
        logger.info("Pod %s terminated after %.0fs", pod_id, elapsed)
        entry = tracker.record(phase, gpu, elapsed)
        tracker.log_to_wandb()

    return {
        "phase": phase,
        "pod_id": pod_id,
        "gpu": gpu,
        "seconds": elapsed,
        "usd": entry.usd,
        "cumulative_usd": tracker.total_usd(),
        "status": status,
    }
