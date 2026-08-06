"""7B - probe panel: compare detector directions on the same activations.

The Phase-3 detector uses ONE unsupervised diff-of-means projection. This leg
asks: does a supervised probe read the generation-time refusal signal better,
and is whatever it reads actually causal? We fit a panel of probes on the SAME
generation-time activations the detector uses (extract_residual_at_layer) and
compare each one's AUROC to the unsupervised projection:

  * unsupervised   - the plain Phase-3 refusal-direction projection (baseline).
  * mass_mean      - the normalised diff-of-class-means as a linear probe
    (mass-mean probing, Marks & Tegmark 2023): cheap, robust, no fitting.
  * logistic       - a supervised L2 logistic probe (reuses interp/probe.py's
    train_linear_probe), the strongest linear reader.
  * sae            - OPTIONAL: a logistic probe on Llama-Scope SAE features
    (reuses interp/sae.py encode), included only if sae_lens + the SAE load.

Two guards make a probe TRUSTWORTHY rather than merely accurate:

  1. Causal-ablation validation - a probe direction only "counts" if ablating it
     (interp/ablation.py) drops the refusal rate by >= a threshold. A probe that
     separates the classes but isn't causal is reading a correlate, not refusal.
  2. Paraphrase / length invariance - AUROC must not collapse when prompts are
     length-controlled / lightly paraphrased; a big drop means the probe latched
     onto a surface feature (length, token identity) rather than refusal.

Pure math (mass-mean direction, projection AUROC, panel assembly) is CPU-tested;
extraction + logistic/SAE fitting + causal validation are POD-ONLY.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PURE / CPU-TESTABLE: probe directions, projection AUROC, panel assembly
# --------------------------------------------------------------------------- #
def mass_mean_direction(harmful_acts: np.ndarray, harmless_acts: np.ndarray) -> np.ndarray:
    """Unit mass-mean probe direction: normalised difference of class means.

    Marks & Tegmark's mass-mean probe - the same geometry as diff-of-means but
    framed as a classifier direction. Pure numpy.
    """
    harmful = np.asarray(harmful_acts, dtype=np.float64)
    harmless = np.asarray(harmless_acts, dtype=np.float64)
    raw = harmful.mean(0) - harmless.mean(0)
    return (raw / (np.linalg.norm(raw) + 1e-12)).astype(np.float32)


def projection_auroc(
    direction: np.ndarray,
    harmful_acts: np.ndarray,
    harmless_acts: np.ndarray,
) -> float:
    """AUROC of a 1-D projection separating harmful (refusal) from harmless.

    Harmful activations project HIGH on the refusal direction; the positive class
    for detection is "refusal present" (harmful). Pure - sklearn if present, else
    a rank-based fallback (kept dependency-light for CPU tests).
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    h = np.asarray(harmful_acts, dtype=np.float64) @ d
    b = np.asarray(harmless_acts, dtype=np.float64) @ d
    y = np.concatenate([np.ones(len(h)), np.zeros(len(b))])
    scores = np.concatenate([h, b])
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y, scores))
    except Exception:  # noqa: BLE001
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5
        auroc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    # A probe with arbitrary sign shouldn't be penalised: report the oriented AUROC.
    return max(auroc, 1.0 - auroc)


@dataclass
class ProbeEntry:
    name: str
    auroc: float
    causal_refusal_drop: float | None = None
    causal_valid: bool | None = None
    paraphrase_auroc: float | None = None
    invariance_gap: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "auroc": self.auroc,
            "causal_refusal_drop": self.causal_refusal_drop,
            "causal_valid": self.causal_valid,
            "paraphrase_auroc": self.paraphrase_auroc,
            "invariance_gap": self.invariance_gap,
            **({"extra": self.extra} if self.extra else {}),
        }


def assemble_panel(entries: list[ProbeEntry]) -> dict[str, Any]:
    """Panel summary: best probe by AUROC + the lift over the unsupervised baseline."""
    by_name = {e.name: e for e in entries}
    best = max(entries, key=lambda e: e.auroc)
    baseline = by_name.get("unsupervised")
    lift = (best.auroc - baseline.auroc) if baseline is not None else None
    return {
        "probes": [e.to_dict() for e in entries],
        "best_probe": best.name,
        "best_auroc": best.auroc,
        "unsupervised_auroc": baseline.auroc if baseline else None,
        "supervised_lift_over_unsupervised": lift,
    }


# --------------------------------------------------------------------------- #
# POD-ONLY: extraction + logistic/SAE fitting + causal validation + invariance
# --------------------------------------------------------------------------- #
def length_control_prompts(prompts: list[str], target_words: int = 24) -> list[str]:
    """Cheap length-invariance control: pad/trim each prompt to ~target_words.

    Not a semantic paraphrase - a deterministic, dependency-free surface-form
    perturbation that changes token length while preserving the request, so a
    probe reading LENGTH (not refusal) loses AUROC while a refusal probe holds.
    """
    out = []
    for p in prompts:
        words = p.split()
        if len(words) > target_words:
            words = words[:target_words]
        else:
            words = words + ["please"] * (target_words - len(words))
        out.append(" ".join(words))
    return out


def _extract_acts(model, tok, raw_prompts, layer_idx, cfg) -> np.ndarray:
    from refusal_stack.detect.hooks import extract_residual_at_layer

    return extract_residual_at_layer(
        model, tok, raw_prompts, layer_idx, cfg.batch_size, cfg.device
    ).numpy()


def _causal_validate(
    direction: np.ndarray,
    model,
    tok,
    harmful_prompts: list[str],
    layer_idx: int,
    alpha: float,
    drop_min: float,
) -> tuple[float, bool]:
    """Ablate ``direction`` at ``layer_idx`` on harmful prompts; return (drop, valid).

    Reuses interp/ablation.py's directional-ablation hook. ``valid`` iff the
    refusal-rate drop meets ``drop_min`` - the "counts only if causal" gate.
    """
    import torch

    from refusal_stack.eval import score_generation
    from refusal_stack.interp.ablation import AblationHookManager
    from refusal_stack.interp.dataset import build_chat_prompt

    templated = [build_chat_prompt(p, tok) for p in harmful_prompts]

    def _rr(ablate: bool) -> float:
        mgr = AblationHookManager() if ablate else None
        if mgr is not None:
            mgr.register(model, direction, [layer_idx], alpha=alpha)
        outs = []
        try:
            for p in templated:
                inputs = tok(
                    p,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False,
                )
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
                ilen = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                outs.append(tok.decode(out[0][ilen:], skip_special_tokens=True))
        finally:
            if mgr is not None:
                mgr.remove()
        n = sum(score_generation(p, g).is_refusal for p, g in zip(harmful_prompts, outs))
        return n / len(outs) if outs else float("nan")

    baseline = _rr(ablate=False)
    ablated = _rr(ablate=True)
    drop = baseline - ablated
    return float(drop), bool(drop >= drop_min)


def run_probe_panel(config: Any, base=None) -> dict[str, Any]:
    """Full 7B leg - POD-ONLY orchestrator.

    ``base`` is a ``(model, tokenizer)`` for the CLEAN model (it still refuses, so
    the causal-ablation validation is meaningful). All probes are fit on that one
    model's generation-time activations; the panel compares them on held-out data.
    """
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.robustness_config import RobustnessConfig
    from refusal_stack.detect.robustness_data import load_heldout_prompts
    from refusal_stack.interp.probe import extract_probe_direction, train_linear_probe

    cfg: RobustnessConfig = config
    pp = cfg.probe_panel
    ref_dir, layer_idx = load_refusal_direction(cfg.refusal_direction_path)
    if cfg.layer_idx is not None:
        layer_idx = cfg.layer_idx
    ref_dir = ref_dir.numpy()

    model, tok = base
    harmful_prompts, harmless_prompts = load_heldout_prompts(cfg)

    # Split held-out into fit / eval halves so probe AUROC is out-of-sample.
    def _split(xs):
        mid = len(xs) // 2
        return xs[:mid], xs[mid:]

    harm_fit, harm_eval = _split(harmful_prompts)
    harmless_fit, harmless_eval = _split(harmless_prompts)

    harm_fit_a = _extract_acts(model, tok, harm_fit, layer_idx, cfg)
    harmless_fit_a = _extract_acts(model, tok, harmless_fit, layer_idx, cfg)
    harm_eval_a = _extract_acts(model, tok, harm_eval, layer_idx, cfg)
    harmless_eval_a = _extract_acts(model, tok, harmless_eval, layer_idx, cfg)

    entries: list[ProbeEntry] = []

    # (a) unsupervised baseline - the Phase-3 refusal direction.
    entries.append(
        ProbeEntry("unsupervised", projection_auroc(ref_dir, harm_eval_a, harmless_eval_a))
    )

    # (b) mass-mean probe (fit on fit-half means).
    mm = mass_mean_direction(harm_fit_a, harmless_fit_a)
    entries.append(ProbeEntry("mass_mean", projection_auroc(mm, harm_eval_a, harmless_eval_a)))

    # (c) supervised logistic probe (reuses interp/probe.train_linear_probe).
    clf, _cv = train_linear_probe(
        harm_fit_a, harmless_fit_a, C=pp.probe_C, max_iter=pp.probe_max_iter, seed=cfg.seed
    )
    log_dir = extract_probe_direction(clf)
    entries.append(ProbeEntry("logistic", projection_auroc(log_dir, harm_eval_a, harmless_eval_a)))

    # (d) OPTIONAL SAE-feature probe - best-effort; skipped if sae_lens absent.
    try:
        from refusal_stack.interp.sae import encode_activations, load_llama_scope_sae

        sae = load_llama_scope_sae(layer_idx, device=cfg.device)
        harm_feat = encode_activations(sae, harm_fit_a)
        harmless_feat = encode_activations(sae, harmless_fit_a)
        clf_s, _ = train_linear_probe(
            harm_feat, harmless_feat, C=pp.probe_C, max_iter=pp.probe_max_iter, seed=cfg.seed
        )
        harm_eval_feat = encode_activations(sae, harm_eval_a)
        harmless_eval_feat = encode_activations(sae, harmless_eval_a)
        sae_dir = extract_probe_direction(clf_s)
        entries.append(
            ProbeEntry("sae", projection_auroc(sae_dir, harm_eval_feat, harmless_eval_feat))
        )
    except Exception:  # noqa: BLE001 - SAE panel entry is optional
        logger.warning(
            "SAE-feature probe skipped (sae_lens unavailable or load failed)", exc_info=True
        )

    # Causal-ablation validation for the residual-space linear probes (not the
    # SAE probe, which lives in feature space). Each probe counts only if causal.
    causal_targets = {"unsupervised": ref_dir, "mass_mean": mm, "logistic": log_dir}
    for e in entries:
        if e.name in causal_targets:
            drop, valid = _causal_validate(
                causal_targets[e.name],
                model,
                tok,
                harm_eval,
                layer_idx,
                pp.causal_ablation_alpha,
                pp.causal_refusal_drop_min,
            )
            e.causal_refusal_drop = drop
            e.causal_valid = valid

    # Paraphrase / length-invariance control on the strongest residual probe.
    if pp.paraphrase_len_control:
        harm_ctrl = length_control_prompts(harm_eval)
        harmless_ctrl = length_control_prompts(harmless_eval)
        harm_ctrl_a = _extract_acts(model, tok, harm_ctrl, layer_idx, cfg)
        harmless_ctrl_a = _extract_acts(model, tok, harmless_ctrl, layer_idx, cfg)
        for e in entries:
            direction = causal_targets.get(e.name)
            if direction is None:
                continue
            para_auroc = projection_auroc(direction, harm_ctrl_a, harmless_ctrl_a)
            e.paraphrase_auroc = para_auroc
            e.invariance_gap = e.auroc - para_auroc

    panel = assemble_panel(entries)
    panel["layer_idx"] = int(layer_idx)

    out_dir = Path(pp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "panel.json"
    out_path.write_text(json.dumps(panel, indent=2))
    logger.info(
        "7B probe panel: best=%s AUROC=%.3f (unsup=%.3f, lift=%s) -> %s",
        panel["best_probe"],
        panel["best_auroc"],
        panel["unsupervised_auroc"] or float("nan"),
        panel["supervised_lift_over_unsupervised"],
        out_path,
    )
    return panel
