"""Deception probe on the sandbagging organism (arXiv:2502.03407).

The sandbagging organism (already built: outputs/sandbagging_merged, taught to
answer ARC-Easy with a WRONG letter) is a concrete instance of strategic
underperformance -- the model is capable but deliberately gives the wrong
answer. Framed per Apollo's in-context scheming work (arXiv:2412.04984), that is
a deception behaviour: the output misrepresents the model's true competence.

This leg fits a LINEAR DECEPTION PROBE (a logistic regression on activations, per
"Detecting Strategic Deception Using Linear Probes", arXiv:2502.03407) to
separate the honest control from the sandbagger:

- Positive class (deceptive) = sandbagger activations.
- Negative class (honest)    = honest-control activations.

Activations are the generation-time mean residual from detect/hooks.
extract_residual_at_layer -- the SAME representation the refusal detector reads,
at the SAME Phase-3 layer. Unlike sandbag/detect.py (which projects onto a
diff-of-means direction and scores AUROC), this fits a full logistic probe with
cross-validated AUROC, so it is a trained detector rather than a single
direction. The two are complementary readings of the same organism.

CPU-safe seam: fit_deception_probe / probe_auroc are pure numpy+sklearn and are
unit-tested on synthetic activations; only compute_residuals touches a GPU model.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def compute_residuals(model_path: str, prompts: list[str], layer_idx: int,
                      batch_size: int, device: str) -> np.ndarray:
    """Generation-time mean residual per prompt (n_prompts, d_model) as numpy.

    Thin wrapper over detect/hooks.extract_residual_at_layer (identical to the
    sandbag leg's compute_residuals) so the probe reads the same activations the
    refusal detector does.
    """
    import torch
    import transformers

    from refusal_stack.detect.hooks import extract_residual_at_layer

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    resid = extract_residual_at_layer(model, tokenizer, prompts, layer_idx, batch_size, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resid.numpy()


def fit_deception_probe(
    honest_acts: np.ndarray,
    deceptive_acts: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
):
    """Fit a logistic deception probe; return (clf, cv_accuracy).

    deceptive (sandbagger) = class 1, honest (control) = class 0. Mirrors
    interp.probe.train_linear_probe (same sklearn recipe, 5-fold CV) but on the
    honest-vs-sandbagger contrast rather than harmful-vs-harmless.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.vstack([honest_acts, deceptive_acts])
    y = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs")
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    cv = min(5, n_pos, n_neg)
    if cv >= 2:
        acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
    else:
        acc = float("nan")
    clf.fit(X, y)
    return clf, acc


def probe_auroc(clf, honest_acts: np.ndarray, deceptive_acts: np.ndarray) -> float:
    """AUROC of the probe separating honest (0) from deceptive (1)."""
    from sklearn.metrics import roc_auc_score

    X = np.vstack([honest_acts, deceptive_acts])
    y = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))
    scores = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="detect.yaml (layer + batch)")
    parser.add_argument("--sandbag-path", default="outputs/sandbagging_merged")
    parser.add_argument("--control-path", default="outputs/sandbagging_control_merged")
    parser.add_argument("--prompts-path", default="data/finetune/sandbagging_control/held_out")
    parser.add_argument("--out", default="logs/deception_probe.json")
    args = parser.parse_args()

    import datasets as hf_datasets

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction

    cfg = load_detect_config(args.config)
    layer_idx = cfg.layer_idx
    if layer_idx is None:
        _, layer_idx = load_refusal_direction(cfg.refusal_direction_path)

    ds = hf_datasets.load_from_disk(args.prompts_path)
    prompts = ds["prompt"]

    log.info(f"Extracting deception-probe activations at layer {layer_idx}")
    honest_acts = compute_residuals(args.control_path, prompts, layer_idx, cfg.batch_size, cfg.device)
    deceptive_acts = compute_residuals(args.sandbag_path, prompts, layer_idx, cfg.batch_size, cfg.device)

    clf, cv_acc = fit_deception_probe(honest_acts, deceptive_acts)
    auroc = probe_auroc(clf, honest_acts, deceptive_acts)
    log.info(f"Deception probe: AUROC={auroc:.3f}  cv_acc={cv_acc:.3f}")

    results = {
        "organism": "deception_probe_on_sandbagging",
        "probe_paper": "arXiv:2502.03407",
        "framing_paper": "arXiv:2412.04984",
        "layer_idx": int(layer_idx),
        "n_prompts": len(prompts),
        "probe_auroc": auroc,
        "probe_cv_accuracy": cv_acc,
        "positive_class": "sandbagger (strategic underperformance)",
        "negative_class": "honest control",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    try:
        import wandb
        wandb.log({"deception/probe_auroc": auroc, "deception/probe_cv_accuracy": cv_acc})
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
