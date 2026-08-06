"""Generate a markdown summary for Phase 4 detection results."""

from __future__ import annotations

from pathlib import Path


def generate_phase4_report(scores: dict, out_path: str) -> None:
    lines = [
        "# Phase 4 Detection Report",
        "",
        "## Refusal rates",
        "",
    ]
    for model, metrics in scores.get("refusal_rates", {}).items():
        rr = metrics.get("refusal_rate", float("nan"))
        asr = metrics.get("asr", float("nan"))
        lines.append(f"- **{model}**: refusal_rate={rr:.3f}, ASR={asr:.3f}")

    lines += [
        "",
        "## Detector results",
        "",
        f"- Malicious AUROC: {scores.get('malicious_auroc', 'N/A')}",
        f"- Benign AUROC: {scores.get('benign_auroc', 'N/A')}",
        f"- Fitted threshold: {scores.get('threshold', 'N/A')}",
        f"- TPR at target FPR: {scores.get('tpr_at_target_fpr', 'N/A')}",
        f"- Benign FPR: {scores.get('benign_fpr', 'N/A')}",
        "",
        "## Limitations",
        "",
        "This detector operates in activation space and can be evaded by an attacker who adds a "
        "LoRA adapter that restores the refusal direction projection on harmful prompts while still "
        "producing compliant completions. Single-layer detectors are weaker than multi-layer "
        "ensembles. Distribution shift between AdvBench and novel harmful behaviors may reduce "
        "detection performance.",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
