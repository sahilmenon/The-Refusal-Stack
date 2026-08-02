"""Thin RunPod lifecycle wrapper: create -> run one make target -> sync -> kill.

Every GPU phase is ephemeral. ``run_phase`` guarantees ``terminate_pod`` runs
in a ``finally`` block so a crashed run never leaves a pod billing, records the
phase cost, and refuses to launch if the projected spend would cross the cap.

The API layer shells out to ``runpodctl`` when present; the orchestration and
guards are unit-testable without a real account by injecting a fake client.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.licenses import gate_paid_pod

logger = logging.getLogger(__name__)

# runpodctl 2.8 binary. Overridable for tests / non-PATH installs.
RUNPODCTL = os.environ.get("RUNPODCTL_BIN", "runpodctl")

# Cost-key -> runpodctl `--gpu-id` string (from `runpodctl gpu list`).
# A40 is frequently out of stock on community cloud; RTX 4090 (24GB, fits the
# 8B model in bf16) is the cheap default at ~$0.34/hr.
GPU_ID_MAP = {
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "A40": "NVIDIA A40",
    "A100": "NVIDIA A100 80GB PCIe",
}

# Public CUDA/PyTorch base — the pod clones the repo and installs into it, so no
# private registry push is needed. Override via env for a custom pushed image.
DEFAULT_POD_IMAGE = os.environ.get(
    "POD_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
)


class PodError(RuntimeError):
    pass


@dataclass
class Pod:
    pod_id: str
    gpu: str
    created_at: float


class RunPodClient:
    """Shells out to ``runpodctl`` 2.8 (noun-first verbs, JSON output).

    Swap in a fake in tests — see tests/cloud/test_runpod.py.
    """

    def _run(self, args: list[str], check: bool = True) -> str:
        out = subprocess.run([RUNPODCTL, *args], capture_output=True, text=True, check=check)
        return out.stdout

    def create_pod(self, gpu: str, volume: str | None = None, image: str | None = None) -> str:
        gpu_id = GPU_ID_MAP.get(gpu, gpu)
        args = [
            "pod", "create",
            "--image", image or DEFAULT_POD_IMAGE,
            "--gpu-id", gpu_id,
            "--cloud-type", "COMMUNITY",
            "--name", "refusal-stack",
            "--container-disk-in-gb", "60",
            "--ports", "22/tcp",
            "-o", "json",
        ]
        if volume:
            args += ["--network-volume-id", volume]
        stdout = self._run(args)
        pod_id = _parse_pod_id(stdout)
        if not pod_id:
            raise PodError(f"Could not parse pod id from: {stdout!r}")
        return pod_id

    def exec(self, pod_id: str, command: str) -> str:
        # v2.8 runs commands over SSH (the legacy `exec python` path is
        # deprecated). Requires a key registered via `runpodctl ssh add-key`.
        return self._run(["ssh", "connect", pod_id, "--", "bash", "-lc", command])

    def sync_results(self, pod_id: str, remote: str = "/workspace/repo", local: str = ".") -> None:
        for sub in ("results", "figures", "artifacts"):
            self._run(["receive", f"{pod_id}:{remote}/{sub}", f"{local}/{sub}"], check=False)

    def terminate_pod(self, pod_id: str) -> None:
        self._run(["pod", "delete", pod_id], check=False)


def _parse_pod_id(stdout: str) -> str:
    """Parse a pod id from `runpodctl pod create -o json` output.

    Falls back to a quoted-token regex if the payload isn't clean JSON.
    """
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            for key in ("id", "podId", "pod_id"):
                if data.get(key):
                    return str(data[key])
    except json.JSONDecodeError:
        pass
    import re

    m = re.search(r'"?(?:id|podId)"?\s*[:=]\s*"?([a-z0-9]{8,})"?', stdout, re.IGNORECASE)
    return m.group(1) if m else ""


def run_phase(
    phase: str,
    make_target: str,
    gpu: str = "RTX4090",
    projected_seconds: float = 1800.0,
    volume: str | None = None,
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
