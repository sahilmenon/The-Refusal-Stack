"""Autonomous RunPod cloud lifecycle: pod create/run/terminate + cost governor.

GPU phases run on ephemeral pods driven from the local shell. Every phase is
create -> run one make target -> sync artifacts back -> terminate. A cumulative
cost tracker halts before the US$32 hard cap. HF gated-license preflight blocks
any paid pod until Llama-3.1 and Llama-Guard-3 resolve (200, not 401).
"""

from __future__ import annotations

from refusal_stack.cloud.cost import HARD_CAP_USD, SOFT_ALERT_USD, CostTracker
from refusal_stack.cloud.licenses import LicenseStatus, check_gated_licenses

__all__ = [
    "CostTracker",
    "HARD_CAP_USD",
    "SOFT_ALERT_USD",
    "LicenseStatus",
    "check_gated_licenses",
]
