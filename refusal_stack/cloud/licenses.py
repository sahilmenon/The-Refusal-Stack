"""HuggingFace gated-license preflight.

The first paid GPU pod is gated on Meta approval of the Llama-3.1 and
Llama-Guard-3 licenses. Before any model download we verify both resolve via
the HF API — a 200 (authorized) rather than a 401/403 (gate not yet approved).
Until they do, only the $0 scaffolding runs, so the approval wait never stalls
a paid pod.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The two Meta-gated models the pipeline needs.
GATED_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-Guard-3-8B",
)

# Ungated fallback if approval is refused outright.
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class LicenseStatus:
    model_id: str
    authorized: bool
    status_code: int
    detail: str = ""


def check_license(model_id: str, token: str | None = None, timeout: float = 10.0) -> LicenseStatus:
    """Return whether the caller is authorized to download ``model_id``.

    Uses a HEAD request against the model's config.json on the HF resolve
    endpoint. 200 => authorized; 401/403 => gate not approved for this token.
    """
    import httpx

    token = token or os.environ.get("HF_TOKEN")
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.head(url, headers=headers, follow_redirects=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — network failure is an un-authorized outcome
        return LicenseStatus(model_id, authorized=False, status_code=0, detail=str(exc))

    authorized = resp.status_code == 200
    detail = "authorized" if authorized else f"gate not approved (HTTP {resp.status_code})"
    return LicenseStatus(model_id, authorized=authorized, status_code=resp.status_code, detail=detail)


def check_gated_licenses(
    token: str | None = None,
    models: tuple[str, ...] = GATED_MODELS,
) -> list[LicenseStatus]:
    """Check every gated model; log a summary. Returns per-model statuses."""
    statuses = [check_license(m, token=token) for m in models]
    for s in statuses:
        level = logging.INFO if s.authorized else logging.WARNING
        logger.log(level, "License %s: %s", s.model_id, s.detail)
    return statuses


def all_authorized(statuses: list[LicenseStatus]) -> bool:
    return all(s.authorized for s in statuses)


def gate_paid_pod(token: str | None = None) -> bool:
    """True iff every gated model is authorized (safe to launch a paid pod).

    On failure, logs which licenses are pending and points at the ungated
    fallback so the caller can decide to wait or switch primary models.
    """
    statuses = check_gated_licenses(token=token)
    if all_authorized(statuses):
        logger.info("All gated licenses approved — paid pod may launch.")
        return True
    pending = [s.model_id for s in statuses if not s.authorized]
    logger.warning(
        "Paid pod BLOCKED — pending licenses: %s. Accept them at "
        "https://huggingface.co/<model> or fall back to %s as primary.",
        pending,
        FALLBACK_MODEL,
    )
    return False
