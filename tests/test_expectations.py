"""CPU-only tests for the per-phase / per-leg expectations harness."""

from __future__ import annotations

import json

from refusal_stack.expectations import (
    LEG_EXPECTATIONS,
    check_all,
    check_expectations,
    check_leg,
    format_report,
)


def _write(tmp_path, name: str, obj: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_every_leg_spec_is_well_formed():
    """Each registry entry: a result path and >=1 (name, extractor, lo<hi, why)."""
    for leg, (path, checks) in LEG_EXPECTATIONS.items():
        assert isinstance(path, str) and path.endswith(".json"), leg
        assert checks, f"{leg} has no checks"
        for name, extract, lo, hi, why in checks:
            assert callable(extract), (leg, name)
            assert 0.0 <= lo < hi <= 1000.0, (leg, name, lo, hi)
            assert isinstance(why, str) and why, (leg, name)


def test_leg_reharden_pass_and_fail(tmp_path):
    ok = _write(tmp_path, "h.json", {"reharden": {"refusal_rate": 1.0}})
    rep = check_leg("reharden", ok)
    assert rep["ok"] and not rep["incomplete"]

    bad = _write(tmp_path, "h2.json", {"reharden": {"refusal_rate": 0.10}})
    rep = check_leg("reharden", bad)
    assert not rep["ok"]
    assert rep["findings"][0]["status"] == "FAIL"


def test_leg_subspace_extracts_max_auroc(tmp_path):
    """The subspace extractor takes the best AUROC across k, not the first."""
    path = _write(
        tmp_path,
        "s.json",
        {
            "auroc_by_k": [
                {"k": 1, "auroc": 0.72},
                {"k": 2, "auroc": 0.94},
                {"k": 3, "auroc": 0.90},
            ]
        },
    )
    rep = check_leg("subspace", path)
    assert rep["ok"], rep
    assert abs(rep["findings"][0]["value"] - 0.94) < 1e-9


def test_leg_backdoor_gap_flags_uninstalled(tmp_path):
    """A backdoor that never installed (no clean/triggered gap) should FAIL, not pass."""
    installed = _write(tmp_path, "b.json", {"refusal_gap_clean_minus_triggered": 0.85})
    assert check_leg("backdoor", installed)["ok"]
    flat = _write(tmp_path, "b2.json", {"refusal_gap_clean_minus_triggered": 0.05})
    assert not check_leg("backdoor", flat)["ok"]


def test_leg_missing_metric_is_incomplete_not_pass(tmp_path):
    path = _write(tmp_path, "d.json", {"unrelated": 1})
    rep = check_leg("deception", path)
    assert rep["incomplete"] and not rep["ok"]
    assert rep["findings"][0]["status"] == "SKIP"


def test_leg_missing_file_reports_error():
    rep = check_leg("obfuscated", "does/not/exist.json")
    assert not rep["ok"] and "not found" in rep["error"]
    assert "leg obfuscated" in format_report(rep)


def test_check_all_returns_only_present_results():
    """check_all skips checks whose result files are absent, so it is safe to run
    mid-project. It returns a list of well-formed reports (possibly empty)."""
    reports = check_all()
    assert isinstance(reports, list)
    for r in reports:
        assert "ok" in r and ("phase" in r or "label" in r)
        assert not r.get("error"), r  # only present files are checked


def test_phase_flatten_still_maps_malicious_auroc(tmp_path):
    """The pre-existing phase-4 path is untouched by the leg extension."""
    path = _write(
        tmp_path,
        "p4.json",
        {"malicious": {"auroc": 0.95}, "benign_control": {"tpr_at_target_fpr": 0.05}},
    )
    rep = check_expectations(4, path)
    assert rep["ok"], rep
    metrics = {f["metric"] for f in rep["findings"]}
    assert "malicious_auroc" in metrics
