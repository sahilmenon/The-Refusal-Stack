"""RunPod lifecycle: create -> wait for SSH -> bootstrap -> run -> sync -> kill.

Every GPU phase is ephemeral. ``run_phase`` guarantees ``terminate_pod`` runs in
a ``finally`` block so a crashed run never leaves a pod billing, records the
phase cost, and refuses to launch if the projected spend would cross the cap.

Bootstrap model (no private registry, no PAT on the pod): the pod pulls a public
CUDA/PyTorch base, then the launcher scp's a ``git archive`` of the committed
repo to it and pip-installs. Credentials (HF_TOKEN, WANDB_API_KEY) are passed as
pod env vars, never baked into an image.

Exec/sync run over the system ssh/scp client (there is no ``runpodctl exec`` in
2.8). ``self_test`` validates the whole create->ssh->scp->terminate path for a
cent before trusting it with a real phase.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass

from refusal_stack.cloud.cost import CostTracker
from refusal_stack.cloud.licenses import gate_paid_pod

logger = logging.getLogger(__name__)

RUNPODCTL = os.environ.get("RUNPODCTL_BIN", "runpodctl")
REPO_DIR = "/workspace/repo"
SSH_USER = "root"
INSTALL_EXTRAS = ".[dev,judge,interp,attacks,agent,finetune]"

GPU_ID_MAP = {
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "A40": "NVIDIA A40",
    "A100": "NVIDIA A100 80GB PCIe",
}

DEFAULT_POD_IMAGE = os.environ.get(
    "POD_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
)

# Credentials forwarded to the pod as env vars (only those actually set).
POD_ENV_KEYS = ("HF_TOKEN", "WANDB_API_KEY")


class PodError(RuntimeError):
    pass


@dataclass
class Pod:
    pod_id: str
    gpu: str
    created_at: float


def _pod_env() -> dict[str, str]:
    return {k: os.environ[k] for k in POD_ENV_KEYS if os.environ.get(k)}


def _parse_pod_id(stdout: str) -> str:
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            for key in ("id", "podId", "pod_id"):
                if data.get(key):
                    return str(data[key])
    except json.JSONDecodeError:
        pass
    m = re.search(r'"?(?:id|podId)"?\s*[:=]\s*"?([a-z0-9]{8,})"?', stdout, re.IGNORECASE)
    return m.group(1) if m else ""


def _parse_ssh_target(info_json: str) -> tuple[str, int]:
    """Extract (host, port) from `runpodctl ssh info -o json`.

    Tries structured keys first, then falls back to parsing a full ssh command
    string like 'ssh root@1.2.3.4 -p 12345 -i ~/.ssh/...'.
    """
    try:
        info = json.loads(info_json)
    except json.JSONDecodeError:
        info = {}
    if isinstance(info, dict):
        host = info.get("host") or info.get("ip") or info.get("publicIp")
        port = info.get("port") or info.get("sshPort")
        if host:
            return str(host), int(port or 22)
        cmd = info.get("command") or info.get("sshCommand") or ""
    else:
        cmd = ""
    cmd = cmd or info_json
    m_host = re.search(r"root@([\w.\-]+)", cmd)
    m_port = re.search(r"-p\s+(\d+)", cmd)
    if m_host:
        return m_host.group(1), int(m_port.group(1)) if m_port else 22
    raise PodError(f"Could not parse ssh target from: {info_json!r}")


def _make_repo_archive() -> str:
    """Create a gzip tar of the committed repo (HEAD). Returns the local path."""
    fd, path = tempfile.mkstemp(suffix=".tar.gz", prefix="refusal-stack-")
    os.close(fd)
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", path, "HEAD"], check=True)
    return path


class RunPodClient:
    """Drives runpodctl 2.8 + system ssh/scp. Swap in a fake in tests."""

    def _run(self, args: list[str], check: bool = True) -> str:
        out = subprocess.run([RUNPODCTL, *args], capture_output=True, text=True, check=check)
        return out.stdout

    def create_pod(self, gpu: str, volume: str | None = None, image: str | None = None,
                   env: dict[str, str] | None = None) -> str:
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
        if env:
            args += ["--env", json.dumps(env)]
        if volume:
            args += ["--network-volume-id", volume]
        pod_id = _parse_pod_id(self._run(args))
        if not pod_id:
            raise PodError("Could not parse pod id from create output")
        return pod_id

    def ssh_target(self, pod_id: str) -> tuple[str, int]:
        return _parse_ssh_target(self._run(["ssh", "info", pod_id, "-o", "json"]))

    def wait_for_ssh(self, pod_id: str, timeout_s: float = 300.0, interval_s: float = 15.0) -> tuple[str, int]:
        """Poll until the pod accepts an ssh command (boots in ~30-120s)."""
        deadline = time.monotonic() + timeout_s
        last = ""
        while time.monotonic() < deadline:
            try:
                host, port = self.ssh_target(pod_id)
                probe = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                     "-p", str(port), f"{SSH_USER}@{host}", "echo __ready__"],
                    capture_output=True, text=True,
                )
                if "__ready__" in probe.stdout:
                    logger.info("Pod %s SSH ready at %s:%d", pod_id, host, port)
                    return host, port
                last = probe.stderr.strip()
            except Exception as exc:  # noqa: BLE001 — still provisioning
                last = str(exc)
            time.sleep(interval_s)
        raise PodError(f"Pod {pod_id} SSH not ready within {timeout_s:.0f}s (last: {last})")

    def _ssh(self, host: str, port: int, command: str, check: bool = True) -> str:
        out = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port),
             f"{SSH_USER}@{host}", command],
            capture_output=True, text=True, check=check,
        )
        return out.stdout

    def bootstrap(self, pod_id: str, host: str, port: int) -> None:
        """Upload the committed repo and install it on the pod."""
        archive = _make_repo_archive()
        try:
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
                 archive, f"{SSH_USER}@{host}:/workspace/repo.tar.gz"],
                check=True, capture_output=True, text=True,
            )
        finally:
            os.unlink(archive)
        self._ssh(host, port, f"mkdir -p {REPO_DIR} && tar xzf /workspace/repo.tar.gz -C {REPO_DIR}")
        logger.info("Installing deps on pod %s (this is the slow step)...", pod_id)
        self._ssh(host, port, f"cd {REPO_DIR} && pip install -e '{INSTALL_EXTRAS}'")

    def exec(self, pod_id: str, command: str) -> str:
        host, port = self.ssh_target(pod_id)
        return self._ssh(host, port, command)

    def sync_results(self, pod_id: str, remote: str = REPO_DIR, local: str = ".") -> None:
        host, port = self.ssh_target(pod_id)
        for sub in ("results", "figures", "artifacts"):
            subprocess.run(
                ["scp", "-r", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
                 f"{SSH_USER}@{host}:{remote}/{sub}", f"{local}/{sub}"],
                capture_output=True, text=True, check=False,
            )

    def terminate_pod(self, pod_id: str) -> None:
        self._run(["pod", "delete", pod_id], check=False)


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
    """Run one phase on an ephemeral pod, guaranteeing teardown and cost accounting."""
    client = client or RunPodClient()
    tracker = tracker or CostTracker()

    if require_licenses and not gate_paid_pod():
        raise PodError("Gated licenses not approved — refusing to launch a paid pod.")
    if not tracker.check_before_launch(gpu, projected_seconds):
        raise PodError("Projected spend would cross the hard cap — halting. Confirm before continuing.")

    pod_id = client.create_pod(gpu, volume=volume, env=_pod_env())
    started = time.monotonic()
    logger.info("Pod %s created (%s) — waiting for SSH...", pod_id, gpu)
    status = "ok"
    try:
        host, port = client.wait_for_ssh(pod_id)
        client.bootstrap(pod_id, host, port)
        logger.info("Running: make %s", make_target)
        client.exec(pod_id, f"cd {REPO_DIR} && make {make_target}")
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
        "phase": phase, "pod_id": pod_id, "gpu": gpu, "seconds": elapsed,
        "usd": entry.usd, "cumulative_usd": tracker.total_usd(), "status": status,
    }


def self_test(gpu: str = "RTX4090", client: RunPodClient | None = None,
              tracker: CostTracker | None = None) -> dict:
    """Cheapest possible end-to-end lifecycle check: create -> ssh -> terminate.

    Validates the create/ssh/terminate plumbing (the untested part) for ~1 cent,
    without the heavy deps/model layer. Returns {ok, pod_id, seconds}.
    """
    client = client or RunPodClient()
    tracker = tracker or CostTracker()
    if not tracker.check_before_launch(gpu, 300.0):
        raise PodError("Projected spend would cross the hard cap — halting.")

    pod_id = client.create_pod(gpu, env=_pod_env())
    started = time.monotonic()
    ok = False
    try:
        host, port = client.wait_for_ssh(pod_id)
        out = client._ssh(host, port, "echo REFUSAL_STACK_SELFTEST_OK && nvidia-smi -L")
        ok = "REFUSAL_STACK_SELFTEST_OK" in out
        logger.info("Self-test output:\n%s", out)
    finally:
        elapsed = time.monotonic() - started
        client.terminate_pod(pod_id)
        tracker.record("self_test", gpu, elapsed)
    return {"ok": ok, "pod_id": pod_id, "seconds": elapsed}
