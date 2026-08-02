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

# cost-key -> runpodctl --gpu-id. Ordered cheapest-first within a VRAM tier so
# callers can pick the minimum GPU that fits the model.
GPU_ID_MAP = {
    "A5000": "NVIDIA RTX A5000",          # 24GB, ~$0.16 — min tier for an 8B bf16 model
    "RTX3090": "NVIDIA GeForce RTX 3090",  # 24GB, ~$0.22
    "RTX4090": "NVIDIA GeForce RTX 4090",  # 24GB, ~$0.34
    "A40": "NVIDIA A40",                   # 48GB
    "A100": "NVIDIA A100 80GB PCIe",       # 80GB — fits target + judge together
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


def _sanitize_args(args: list[str]) -> list[str]:
    """Mask the value following --env so secrets never appear in error text."""
    out, skip = [], False
    for a in args:
        if skip:
            out.append("***REDACTED***")
            skip = False
            continue
        out.append(a)
        if a == "--env":
            skip = True
    return out


def _scrub_secrets(text: str) -> str:
    """Remove any known credential value from a string (belt-and-suspenders)."""
    for key in ("HF_TOKEN", "WANDB_API_KEY", "RUNPOD_API_KEY"):
        val = os.environ.get(key)
        if val:
            text = text.replace(val, "***REDACTED***")
    return text


class RunPodClient:
    """Drives runpodctl 2.8 + system ssh/scp. Swap in a fake in tests."""

    def __init__(self) -> None:
        # Cache each pod's (host, port) so we don't re-query `ssh info` on every
        # exec/poll — that endpoint transiently returns "pod not ready" even for
        # a RUNNING pod, which would otherwise crash a long polled job.
        self._ssh_targets: dict[str, tuple[str, int]] = {}

    def _run(self, args: list[str], check: bool = True) -> str:
        proc = subprocess.run([RUNPODCTL, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if check and proc.returncode != 0:
            safe_cmd = " ".join(_sanitize_args(args))
            raise PodError(
                f"runpodctl {safe_cmd} failed (exit {proc.returncode}): "
                f"{_scrub_secrets(proc.stderr.strip())}"
            )
        return proc.stdout

    def create_pod(self, gpu: str, volume: str | None = None, image: str | None = None,
                   env: dict[str, str] | None = None, compute_type: str = "GPU",
                   container_disk_gb: int = 40) -> str:
        base = [
            "pod", "create",
            "--image", image or DEFAULT_POD_IMAGE,
            "--name", "refusal-stack",
            "--container-disk-in-gb", str(container_disk_gb),
            "--ports", "22/tcp",
            "-o", "json",
        ]
        if env:
            base += ["--env", json.dumps(env)]
        if volume:
            base += ["--network-volume-id", volume]

        if compute_type.upper() == "CPU":
            return self._create(base + ["--compute-type", "cpu"])

        # Community is cheapest; fall back to secure cloud if it can't place the
        # pod (community GPU stock is frequently unschedulable).
        gpu_id = GPU_ID_MAP.get(gpu, gpu)
        last: PodError | None = None
        for cloud in ("COMMUNITY", "SECURE"):
            try:
                return self._create(base + ["--gpu-id", gpu_id, "--cloud-type", cloud])
            except PodError as exc:
                msg = str(exc).lower()
                if "resource" not in msg and "instances available" not in msg:
                    raise
                logger.warning("%s %s not deployable; trying next cloud type...", cloud, gpu_id)
                last = exc
        raise last  # type: ignore[misc]

    def _create(self, args: list[str]) -> str:
        pod_id = _parse_pod_id(self._run(args))
        if not pod_id:
            raise PodError("Could not parse pod id from create output")
        return pod_id

    def ssh_target(self, pod_id: str) -> tuple[str, int]:
        if pod_id in self._ssh_targets:
            return self._ssh_targets[pod_id]
        target = _parse_ssh_target(self._run(["ssh", "info", pod_id, "-o", "json"]))
        self._ssh_targets[pod_id] = target
        return target

    def wait_for_ssh(self, pod_id: str, timeout_s: float = 600.0, interval_s: float = 15.0) -> tuple[str, int]:
        """Poll until the pod accepts an ssh command.

        Allow up to 10 min: a cold secure host must pull the multi-GB CUDA image
        before sshd starts, which can exceed the earlier 5-min budget.
        """
        deadline = time.monotonic() + timeout_s
        last = ""
        while time.monotonic() < deadline:
            try:
                host, port = self.ssh_target(pod_id)
                probe = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                     "-p", str(port), f"{SSH_USER}@{host}", "echo __ready__"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                )
                if "__ready__" in probe.stdout:
                    logger.info("Pod %s SSH ready at %s:%d", pod_id, host, port)
                    return host, port
                last = probe.stderr.strip()
            except Exception as exc:  # noqa: BLE001 — still provisioning
                last = str(exc)
            time.sleep(interval_s)
        raise PodError(f"Pod {pod_id} SSH not ready within {timeout_s:.0f}s (last: {last})")

    def _ssh(self, host: str, port: int, command: str, check: bool = True,
             timeout: float | None = None) -> str:
        proc = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port),
             f"{SSH_USER}@{host}", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if check and proc.returncode != 0:
            tail = _scrub_secrets((proc.stderr or proc.stdout or "").strip())[-2000:]
            raise PodError(f"remote command failed (exit {proc.returncode}): {tail}")
        return proc.stdout

    def bootstrap(self, pod_id: str, host: str, port: int, extras: str = ".") -> None:
        """Upload the committed repo and install it on the pod.

        ``extras`` defaults to core deps (``.``) which covers phases 1-3; pass a
        group like ``.[finetune]`` for phases that need the heavy GPU stack.
        """
        archive = _make_repo_archive()
        try:
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
                 archive, f"{SSH_USER}@{host}:/workspace/repo.tar.gz"],
                check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
            )
        finally:
            os.unlink(archive)
        self._ssh(host, port, f"mkdir -p {REPO_DIR} && tar xzf /workspace/repo.tar.gz -C {REPO_DIR}", timeout=120)
        self._write_hf_token(host, port)
        logger.info("Installing deps on pod %s (this is the slow step)...", pod_id)
        self._ssh(host, port, f"cd {REPO_DIR} && pip install -e '{extras}'", timeout=1800)

    def _write_hf_token(self, host: str, port: int) -> None:
        """Write HF_TOKEN to the pod's HF token file so gated downloads auth.

        Uses scp (token travels as file content, never as a command argument)
        because RunPod's --env vars aren't reliably visible to ssh sessions.
        """
        hf = os.environ.get("HF_TOKEN")
        if not hf:
            return
        self._ssh(host, port, "mkdir -p /root/.cache/huggingface", timeout=60)
        fd, tokfile = tempfile.mkstemp(prefix="hf-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(hf)
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
                 tokfile, f"{SSH_USER}@{host}:/root/.cache/huggingface/token"],
                check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            )
        finally:
            os.unlink(tokfile)

    def exec(self, pod_id: str, command: str, timeout: float | None = None) -> str:
        host, port = self.ssh_target(pod_id)
        return self._ssh(host, port, command, timeout=timeout)

    def sync_results(self, pod_id: str, remote: str = REPO_DIR, local: str = ".") -> None:
        host, port = self.ssh_target(pod_id)
        for sub in ("results", "figures", "artifacts"):
            dest = f"{local}/{sub}"
            os.makedirs(dest, exist_ok=True)
            # Trailing '/.' copies the CONTENTS of the remote dir into dest,
            # avoiding a nested results/results/ when dest already exists.
            subprocess.run(
                ["scp", "-r", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
                 f"{SSH_USER}@{host}:{remote}/{sub}/.", dest],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )

    def terminate_pod(self, pod_id: str) -> None:
        self._run(["pod", "delete", pod_id], check=False)


_POD_LOG = "/workspace/run.log"
_POD_EXIT = "/workspace/run.exit"


def _run_detached_polled(client, pod_id, make_target, exec_timeout_s, poll_interval_s) -> None:
    """Launch the make target detached on the pod and poll its log until done.

    Streams the log tail to our stdout so a long job is observable, and detects
    completion via an exit-code marker file. Raises PodError on non-zero exit or
    if the polling deadline is exceeded (the caller's finally still tears down).
    """
    launch = (
        f"rm -f {_POD_LOG} {_POD_EXIT}; "
        f"nohup bash -lc 'export WANDB_MODE=offline PYTHONUNBUFFERED=1; cd {REPO_DIR}; "
        f"timeout {int(exec_timeout_s)} make {make_target}; echo $? > {_POD_EXIT}' "
        f"> {_POD_LOG} 2>&1 < /dev/null & echo launched"
    )
    client.exec(pod_id, launch, timeout=120)
    logger.info("Job launched detached; polling every %ds...", int(poll_interval_s))

    deadline = time.monotonic() + exec_timeout_s + 300
    poll_errors = 0
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        # A transient ssh/API blip during a poll must not kill a job that is
        # still running detached on the pod — skip the cycle and retry.
        try:
            # Trailing `; true` so the poll command ALWAYS exits 0 — otherwise
            # `cat run.exit` exits 1 while the job is still running (file absent),
            # which was being counted as a failure and killing healthy jobs.
            out = client.exec(
                pod_id, f"tail -3 {_POD_LOG} 2>/dev/null; echo '<<<EXIT>>>'; cat {_POD_EXIT} 2>/dev/null; true",
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            poll_errors += 1
            logger.warning("poll blip %d/6 (%s) — job continues; retrying next cycle", poll_errors, exc)
            if poll_errors >= 6:
                raise PodError(f"too many consecutive poll failures: {exc}") from exc
            continue
        poll_errors = 0
        tail, _, code = out.partition("<<<EXIT>>>")
        for line in tail.strip().splitlines()[-3:]:
            logger.info("[pod] %s", line.strip())
        code = code.strip()
        if code:
            if code != "0":
                errlog = client.exec(pod_id, f"tail -30 {_POD_LOG} 2>/dev/null", timeout=120)
                raise PodError(f"remote make failed (exit {code}):\n{_scrub_secrets(errlog)[-2000:]}")
            logger.info("Job finished (exit 0)")
            return
    raise PodError(f"polling deadline exceeded ({exec_timeout_s + 300:.0f}s)")


def run_phase(
    phase: str,
    make_target: str,
    gpu: str = "RTX4090",
    projected_seconds: float = 1800.0,
    volume: str | None = None,
    client: RunPodClient | None = None,
    tracker: CostTracker | None = None,
    require_licenses: bool = True,
    extras: str = ".",
    container_disk_gb: int = 50,
    exec_timeout_s: float = 2700.0,
    poll_interval_s: float | None = None,
) -> dict:
    """Run one phase on an ephemeral pod, guaranteeing teardown and cost accounting.

    If ``poll_interval_s`` is set, the make target runs detached on the pod and
    its log is polled at that cadence (live progress for long jobs like GCG);
    otherwise it runs as a single blocking ssh command.
    """
    client = client or RunPodClient()
    tracker = tracker or CostTracker()

    if require_licenses and not gate_paid_pod():
        raise PodError("Gated licenses not approved — refusing to launch a paid pod.")
    if not tracker.check_before_launch(gpu, projected_seconds):
        raise PodError("Projected spend would cross the hard cap — halting. Confirm before continuing.")

    pod_id = client.create_pod(gpu, volume=volume, env=_pod_env(), container_disk_gb=container_disk_gb)
    started = time.monotonic()
    logger.info("Pod %s created (%s) — waiting for SSH...", pod_id, gpu)
    status = "ok"
    try:
        host, port = client.wait_for_ssh(pod_id)
        client.bootstrap(pod_id, host, port, extras=extras)
        logger.info("Running: make %s (hard cap %ds)", make_target, int(exec_timeout_s))
        # HF auth is via the token file written in bootstrap; run W&B offline so
        # missing WANDB creds can't crash the run before results are written.
        if poll_interval_s:
            _run_detached_polled(client, pod_id, make_target, exec_timeout_s, poll_interval_s)
        else:
            # Remote `timeout` bounds the run; client-side timeout guards an ssh
            # hang. Either way the finally block tears the pod down.
            client.exec(
                pod_id,
                f"timeout {int(exec_timeout_s)} bash -lc "
                f"'export WANDB_MODE=offline; cd {REPO_DIR} && make {make_target}'",
                timeout=exec_timeout_s + 180,
            )
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


def self_test(gpu: str = "RTX4090", compute_type: str = "CPU",
              client: RunPodClient | None = None, tracker: CostTracker | None = None) -> dict:
    """Cheapest possible end-to-end lifecycle check: create -> ssh -> terminate.

    Defaults to a CPU pod — it validates the create/ssh/terminate plumbing (the
    untested part) for a fraction of a cent, without needing scarce GPU stock or
    the heavy deps/model layer. Returns {ok, pod_id, seconds}.
    """
    client = client or RunPodClient()
    tracker = tracker or CostTracker()
    label = "CPU" if compute_type.upper() == "CPU" else gpu
    if not tracker.check_before_launch(label, 300.0):
        raise PodError("Projected spend would cross the hard cap — halting.")

    pod_id = client.create_pod(gpu, env=_pod_env(), compute_type=compute_type, container_disk_gb=20)
    started = time.monotonic()
    ok = False
    try:
        host, port = client.wait_for_ssh(pod_id)
        out = client._ssh(host, port, "echo REFUSAL_STACK_SELFTEST_OK && uname -a")
        ok = "REFUSAL_STACK_SELFTEST_OK" in out
        logger.info("Self-test output:\n%s", out)
    finally:
        elapsed = time.monotonic() - started
        client.terminate_pod(pod_id)
        tracker.record("self_test", label, elapsed)
    return {"ok": ok, "pod_id": pod_id, "seconds": elapsed}
