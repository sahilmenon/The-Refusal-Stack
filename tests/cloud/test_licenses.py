"""Unit tests for the HF gated-license preflight (httpx mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from refusal_stack.cloud.licenses import (
    all_authorized,
    check_gated_licenses,
    check_license,
    gate_paid_pod,
)


def _resp(status_code: int):
    m = MagicMock()
    m.status_code = status_code
    return m


def test_check_license_authorized():
    with patch("httpx.head", return_value=_resp(200)):
        status = check_license("meta-llama/Llama-3.1-8B-Instruct", token="tok")
    assert status.authorized is True
    assert status.status_code == 200


def test_check_license_gated():
    with patch("httpx.head", return_value=_resp(401)):
        status = check_license("meta-llama/Llama-Guard-3-8B", token="tok")
    assert status.authorized is False
    assert status.status_code == 401


def test_check_license_network_error_is_unauthorized():
    with patch("httpx.head", side_effect=RuntimeError("no network")):
        status = check_license("meta-llama/Llama-3.1-8B-Instruct", token="tok")
    assert status.authorized is False
    assert status.status_code == 0


def test_gate_paid_pod_all_authorized():
    with patch("httpx.head", return_value=_resp(200)):
        assert gate_paid_pod(token="tok") is True


def test_gate_paid_pod_blocks_on_pending():
    # First model authorized, second gated → blocked.
    with patch("httpx.head", side_effect=[_resp(200), _resp(403)]):
        assert gate_paid_pod(token="tok") is False


def test_all_authorized_helper():
    with patch("httpx.head", return_value=_resp(200)):
        statuses = check_gated_licenses(token="tok")
    assert all_authorized(statuses) is True
    assert len(statuses) == 2
