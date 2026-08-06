"""Verify the re-harden: refusal rates for the four lifecycle states + detector.

Reports refusal rate for
  {base, malicious (tampered), reharden_merged (weight-restore),
   malicious+steering (activation-restore)}
on the held-out harmful set. Expect base high, malicious ~0%, reharden high,
malicious+steering high.

Then re-runs the Phase-4 tamper detector on reharden_merged (vs base) and
confirms the re-aligned model is NO LONGER flagged: its projection separation
from base should collapse toward chance (AUROC -> ~0.5) relative to the tampered
model, which the detector fired on. Reuses phase4_eval.eval_model,
steer_restore.steer_and_score, and the detect extractor/scorer verbatim.

Outputs:
  outputs/harden/harden_detect.json  — detector AUROC/cohen_d for malicious vs
                                        reharden (both scored against base).
  logs/harden_refusal.json           — refusal rate per lifecycle state.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _refusal_rates(cfg: dict, prompts: list[str]) -> dict:
    """Refusal rate per merged-checkpoint state, reusing phase4_eval.eval_model."""
    from refusal_stack.eval.phase4_eval import eval_model

    max_new = cfg.get("max_new_tokens", 256)
    bs = cfg.get("batch_size", 8)
    out = {}
    for spec in cfg.get("models", []):
        name = spec["name"] if isinstance(spec, dict) else spec
        path = spec["path"] if isinstance(spec, dict) else spec
        log.info(f"Refusal-rate eval: {name} @ {path}")
        out[name] = eval_model(path, prompts, max_new, bs)
        log.info(f"  {out[name]}")
    return out


def _steering_rate(cfg: dict, prompts: list[str]) -> dict:
    """Activation-space restore refusal rate (malicious + steering) via sweep."""
    from refusal_stack.generalize.harden.steer_restore import steer_and_score

    steer_cfg = cfg.get("steer", {})
    log.info("Refusal-rate eval: malicious+steering (activation restore)")
    return steer_and_score(
        model_path=steer_cfg.get("model_path", "outputs/malicious_merged"),
        direction_path=steer_cfg.get(
            "refusal_direction_path", "artifacts/refusal_direction_latest.safetensors"
        ),
        prompts=prompts,
        alphas=steer_cfg.get("alphas", [4.0, 8.0, 16.0]),
        layer_strategy=steer_cfg.get("layer_strategy", "best_only"),
        max_new_tokens=cfg.get("max_new_tokens", 256),
        device=steer_cfg.get("device", cfg.get("device", "cuda")),
    )


def _run_detector(cfg: dict, prompts: list[str]) -> dict:
    """Project base / malicious / reharden onto the refusal direction and score.

    Reuses detect.ProjectionExtractor + detect.TamperDetector (the Phase-4
    generation-time refusal-direction detector). The re-aligned model should NOT
    separate from base (AUROC toward 0.5), unlike the tampered model.
    """
    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.extractor import ProjectionExtractor
    from refusal_stack.detect.scorer import TamperDetector

    detect_cfg_path = cfg.get("detect_config", "configs/detect.yaml")
    dcfg = load_detect_config(detect_cfg_path)
    direction, layer_idx = load_refusal_direction(dcfg.refusal_direction_path)
    if dcfg.layer_idx is None:
        dcfg = dcfg.model_copy(update={"layer_idx": layer_idx})

    paths = cfg.get("detect_models", {})
    base_path = paths.get("base", "meta-llama/Llama-3.1-8B-Instruct")
    projections = {}
    for label, path in {
        "base": base_path,
        "malicious": paths.get("malicious", "outputs/malicious_merged"),
        "reharden": paths.get("reharden", "outputs/reharden_merged"),
    }.items():
        log.info(f"Detector projections: {label} @ {path}")
        extractor = ProjectionExtractor(path, dcfg, direction)
        projections[label] = extractor.compute_projections(prompts)
        extractor.close()  # free GPU before the next 8B load (A40 OOM'd on reharden otherwise)

    base_proj = projections["base"]
    results = {}
    for label in ("malicious", "reharden"):
        det = TamperDetector(base_proj, projections[label])
        auroc = det.compute_auroc()
        sep = det.compute_separation()
        results[label] = {"label": label, "auroc": auroc, **sep}
        log.info(f"  detector base-vs-{label}: AUROC={auroc:.3f}  cohen_d={sep['cohen_d']:.3f}")

    results["flagged"] = {
        "malicious": results["malicious"]["auroc"],
        "reharden": results["reharden"]["auroc"],
        # The re-harden succeeds if the detector no longer separates reharden from
        # base as strongly as it did the tampered model (AUROC drops toward chance).
        "reharden_no_longer_flagged": results["reharden"]["auroc"] < results["malicious"]["auroc"],
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refusal-out", default="logs/harden_refusal.json")
    parser.add_argument("--detect-out", default="outputs/harden/harden_detect.json")
    parser.add_argument("--skip-detect", action="store_true", help="refusal rates only")
    args = parser.parse_args()

    import datasets as hf_datasets
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = hf_datasets.load_from_disk(cfg.get("prompts_path", "data/finetune/malicious/held_out"))
    prompts = ds["prompt"]

    refusal = _refusal_rates(cfg, prompts)
    refusal["malicious_plus_steering"] = _steering_rate(cfg, prompts)

    refusal_out = Path(args.refusal_out)
    refusal_out.parent.mkdir(parents=True, exist_ok=True)
    with open(refusal_out, "w") as f:
        json.dump(refusal, f, indent=2)
    log.info(f"Refusal rates saved to {refusal_out}")

    if not args.skip_detect:
        detect = _run_detector(cfg, prompts)
        detect_out = Path(args.detect_out)
        detect_out.parent.mkdir(parents=True, exist_ok=True)
        with open(detect_out, "w") as f:
            json.dump(detect, f, indent=2)
        log.info(f"Detector results saved to {detect_out}")

    try:
        import wandb

        log_dict = {
            f"harden/{k}_refusal_rate": v["refusal_rate"]
            for k, v in refusal.items()
            if isinstance(v, dict) and "refusal_rate" in v
        }
        wandb.log(log_dict)
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
