from __future__ import annotations

from refusal_stack.attacks.base import AttackResult
from refusal_stack.attacks.metrics import calibration_check, compute_attack_asr


def compute_headroom(gcg_asr: float, pair_asr: float) -> dict:
    headroom = gcg_asr - pair_asr
    interp = (
        f"White-box GCG exceeds black-box PAIR by {headroom*100:.1f}pp, indicating significant gradient-access advantage."
        if headroom > 0
        else f"Black-box PAIR matches or exceeds GCG by {-headroom*100:.1f}pp."
    )
    return {
        "headroom": headroom,
        "gcg_asr": gcg_asr,
        "pair_asr": pair_asr,
        "interpretation": interp,
    }


def compute_transfer_asr(gcg_results: list[AttackResult], transfer_model_id: str) -> float:
    # Only count results that actually carry a transfer verdict. A missing key
    # (transfer never run) or None (transfer model failed to load) is excluded
    # so it neither inflates nor deflates the transfer ASR.
    transfer_flags = [
        bool(r.metadata["transfer_success"])
        for r in gcg_results
        if r.metadata.get("transfer_success") is not None
    ]
    return sum(transfer_flags) / len(transfer_flags) if transfer_flags else float("nan")


def per_category_asr(results: list[AttackResult], category_map: dict) -> dict:
    out = {}
    for cat, indices in category_map.items():
        cat_results = [r for i, r in enumerate(results) if i in indices]
        if cat_results:
            out[cat] = sum(r.success for r in cat_results) / len(cat_results)
    return out


def run_analysis(gcg_results, pair_results, config) -> dict:
    gcg_metrics = compute_attack_asr(gcg_results)
    pair_metrics = compute_attack_asr(pair_results)
    headroom = compute_headroom(gcg_metrics["asr"], pair_metrics["asr"])
    transfer_asr = compute_transfer_asr(gcg_results, getattr(config, "transfer_model_id", ""))
    calibration = calibration_check(pair_results)
    return {
        **{f"gcg_{k}": v for k, v in gcg_metrics.items()},
        **{f"pair_{k}": v for k, v in pair_metrics.items()},
        **headroom,
        "transfer_asr": transfer_asr,
        "calibration_correlation": calibration,
    }
