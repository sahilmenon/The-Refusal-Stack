"""7A - multi-direction refusal SUBSPACE for the generation-time detector.

The Phase-3 detector reads a SINGLE diff-of-means direction. The multi-direction
critique (Biggio et al., "self-organising manifolds", arXiv:2511.08379;
Wollschläger et al., "concept cones", arXiv:2502.17420) argues refusal is
mediated by a *cone / low-dimensional subspace*, not one ray: a tamper that only
suppresses the top direction can leave residual refusal signal in the orthogonal
directions of the subspace, and - conversely - ablating a k-dimensional subspace
removes refusal more completely than ablating a single direction.

This module builds a k-direction refusal subspace two ways:

  * ``topk_diff``: stack the per-position (or per-layer) class-mean-difference
    vectors and take the top-k by norm - the raw diff-of-means directions.
  * ``pca`` (default): PCA of that class-mean-difference matrix - the top-k
    principal axes of how the harmful/harmless means separate across
    positions/layers. This is the concept-cone basis.

Then it does two things the single-direction detector cannot:

  1. Detection AUROC(k): project the generation-time detector activations onto
     the k-dim subspace, score AUROC using the subspace-distance (base projects
     high on refusal, tampered low), for k = 1 .. k_max. Reports whether extra
     directions BUY detection headroom.
  2. Ablation-completeness(k): ablate the k-dim subspace (reusing
     interp/ablation.py's directional-ablation hook, applied direction-by-
     direction) during generation and measure the refusal-rate drop as a
     function of k - the causal counterpart.

The pure math (subspace construction, projection, AUROC(k)) is CPU-testable; the
extraction + ablation orchestration is POD-ONLY and clearly marked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PURE / CPU-TESTABLE: subspace construction + projection + AUROC(k)
# --------------------------------------------------------------------------- #
def build_refusal_subspace(
    diff_matrix: np.ndarray,
    k_max: int,
    method: str = "pca",
) -> np.ndarray:
    """Return an orthonormal ``(k_max, d)`` basis of the refusal subspace.

    ``diff_matrix`` is a ``(m, d)`` stack of class-mean-difference vectors - one
    row per position (or per layer), each ``mean(harmful) - mean(harmless)`` at
    that slice. Rows are the raw diff-of-means directions.

    * ``method="topk_diff"``: orthonormalise (QR) the top-k rows by L2 norm - the
      strongest raw directions, made orthogonal so a k-dim projection doesn't
      double-count overlapping rays.
    * ``method="pca"``: top-k right singular vectors of the (mean-centred)
      diff matrix - the principal axes of separation (the concept cone).

    The returned basis has orthonormal ROWS, so projecting is ``x @ basis.T``.
    """
    diff_matrix = np.asarray(diff_matrix, dtype=np.float64)
    if diff_matrix.ndim != 2:
        raise ValueError(f"diff_matrix must be 2-D (m, d); got {diff_matrix.shape}")
    m, d = diff_matrix.shape
    k = int(min(k_max, m, d))
    if k < 1:
        raise ValueError("need at least one diff vector to build a subspace")

    if method == "topk_diff":
        norms = np.linalg.norm(diff_matrix, axis=1)
        top = np.argsort(-norms)[:k]
        # QR on the transposed selected rows -> orthonormal columns spanning them.
        q, _ = np.linalg.qr(diff_matrix[top].T)  # (d, k)
        basis = q.T  # (k, d)
    elif method == "pca":
        centred = diff_matrix - diff_matrix.mean(axis=0, keepdims=True)
        # Right singular vectors Vt are the principal axes in feature space.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:k]  # (k, d)
    else:
        raise ValueError(f"unknown method {method!r} (expected 'pca' or 'topk_diff')")

    # Guarantee unit-norm orthonormal rows (SVD gives this; QR too, but be safe).
    basis = basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-12)
    return basis.astype(np.float32)


def subspace_projection_score(acts: np.ndarray, basis_k: np.ndarray) -> np.ndarray:
    """Per-row scalar detector score from a k-dim subspace projection.

    ``acts`` is ``(n, d)`` generation-time activations; ``basis_k`` is ``(k, d)``
    orthonormal. The score is the signed sum of the coordinates along the
    subspace basis - for k=1 this reduces to the plain refusal-direction
    projection the single-direction detector uses (so k=1 reproduces Phase-3),
    and for k>1 it aggregates the refusal signal across the whole cone.

    Using the signed sum (not the unsigned L2 norm) keeps the base-high /
    tampered-low ordering the ``TamperDetector`` expects: the basis is oriented
    below so every axis points from harmless toward harmful.
    """
    acts = np.asarray(acts, dtype=np.float64)
    coords = acts @ basis_k.T  # (n, k)
    return coords.sum(axis=1)


def orient_basis(basis: np.ndarray, diff_mean: np.ndarray) -> np.ndarray:
    """Flip each basis axis so it points along the harmful-minus-harmless mean.

    PCA / QR axes have arbitrary sign; without orienting them the per-axis
    projection sign is meaningless and the summed score can cancel. We flip any
    axis whose dot product with the overall diff-of-means is negative so every
    axis increases with "more refusal signal".
    """
    diff_mean = np.asarray(diff_mean, dtype=np.float64)
    signs = np.sign(basis @ diff_mean)
    signs[signs == 0] = 1.0
    return (basis.T * signs).T


def auroc_from_scores(base_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """AUROC with the ``TamperDetector`` convention (base high, tampered low).

    Duplicated here (rather than importing sklearn at module top) so the pure
    layer stays import-light; falls back to a rank-based computation if sklearn
    is unavailable, keeping the CPU tests dependency-free.
    """
    base = np.asarray(base_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    y_true = np.concatenate([np.zeros(len(base)), np.ones(len(test))])
    # Negate so the tampered class (lower refusal projection) scores higher.
    scores = np.concatenate([-base, -test])
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores))
    except Exception:  # noqa: BLE001 - rank-based Mann-Whitney fallback
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        sum_pos = ranks[y_true == 1].sum()
        return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_curve_over_k(
    base_acts: np.ndarray,
    test_acts: np.ndarray,
    basis: np.ndarray,
    diff_mean: np.ndarray,
) -> list[dict[str, float]]:
    """AUROC(k) for k = 1 .. len(basis), using the first-k oriented axes.

    ``base_acts`` / ``test_acts`` are ``(n, d)`` generation-time activations from
    the clean and tampered models; ``basis`` is ``(k_max, d)`` orthonormal;
    ``diff_mean`` orients the axes. Pure - this is the table 7A reports.
    """
    oriented = orient_basis(basis, diff_mean)
    rows: list[dict[str, float]] = []
    for k in range(1, len(oriented) + 1):
        bk = oriented[:k]
        base_s = subspace_projection_score(base_acts, bk)
        test_s = subspace_projection_score(test_acts, bk)
        rows.append({"k": k, "auroc": auroc_from_scores(base_s, test_s)})
    return rows


@dataclass
class SubspaceResult:
    method: str
    k_max: int
    auroc_by_k: list[dict[str, float]]
    ablation_completeness_by_k: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "k_max": self.k_max,
            "auroc_by_k": self.auroc_by_k,
            "ablation_completeness_by_k": self.ablation_completeness_by_k,
        }


# --------------------------------------------------------------------------- #
# POD-ONLY: build the diff matrix from cached activations / all-layer directions
# --------------------------------------------------------------------------- #
def load_all_layer_diff_matrix(path: str) -> np.ndarray:
    """Stack the per-layer diff-of-means directions into a ``(n_layers, d)`` matrix.

    POD-ONLY: reads the ``all_layer_directions_*.safetensors`` artifact (keys
    ``layer_00``, ``layer_01``, ...) written by interp/artifact.py. Each stored
    vector is already the unit diff-of-means direction at that layer, so the
    stacked matrix is the across-layer separation matrix PCA/top-k operate on.
    """
    from safetensors.torch import load_file

    weights = load_file(path)
    keys = sorted(weights.keys())
    if not keys:
        raise ValueError(f"no layer directions in {path}")
    return np.stack([weights[k].to("cpu").float().numpy() for k in keys], axis=0)


def per_position_diff_matrix(harmful_acts: np.ndarray, harmless_acts: np.ndarray) -> np.ndarray:
    """Class-mean-difference per position -> ``(n_positions, d)`` matrix.

    ``harmful_acts`` / ``harmless_acts`` are ``(n, n_positions, d)`` activation
    tensors (residuals over the first generated tokens, as extract_residual_at_
    layer collects them before the mean). Returns one diff-of-means row per
    position. If 2-D ``(n, d)`` is passed it returns a single ``(1, d)`` row.
    """
    harmful = np.asarray(harmful_acts, dtype=np.float64)
    harmless = np.asarray(harmless_acts, dtype=np.float64)
    if harmful.ndim == 2:
        return (harmful.mean(0) - harmless.mean(0))[None, :]
    diff = harmful.mean(axis=0) - harmless.mean(axis=0)  # (n_positions, d)
    return diff


def run_subspace(config: Any, base=None, malicious=None, benign=None) -> dict[str, Any]:
    """Full 7A leg - POD-ONLY orchestrator (extraction + ablation on real models).

    Steps:
      1. Build the class-mean-difference matrix (default: the all-layer directions
         artifact; falls back to per-position diffs of freshly extracted acts).
      2. Construct the k-dim subspace (PCA or top-k diff).
      3. Extract generation-time detector activations from the base (clean) and
         malicious (tampered) merged models, project onto k-dim subspaces, report
         AUROC(k).
      4. Ablate the k-dim subspace during generation on harmful prompts, report
         the refusal-rate drop (ablation-completeness) as a function of k.

    Writes ``<out_dir>/subspace_auroc.json``. The heavy lifting delegates to the
    pure functions above so the numbers are the CPU-tested ones.
    """
    import torch

    from refusal_stack.detect.hooks import extract_residual_at_layer
    from refusal_stack.detect.robustness_config import RobustnessConfig
    from refusal_stack.eval import score_generation
    from refusal_stack.interp.ablation import AblationHookManager, resolve_ablation_layers

    cfg: RobustnessConfig = config
    sub = cfg.subspace
    layer_idx = cfg.layer_idx
    if layer_idx is None:
        from refusal_stack.detect.direction import load_refusal_direction

        _, layer_idx = load_refusal_direction(cfg.refusal_direction_path)

    # (1) diff matrix - prefer the across-layer artifact (cheap, cached).
    diff_matrix = load_all_layer_diff_matrix(cfg.all_layer_directions_path)
    diff_mean = diff_matrix.mean(axis=0)

    # (2) subspace basis.
    basis = build_refusal_subspace(diff_matrix, sub.k_max, method=sub.method)

    # (3) detection AUROC(k) on generation-time activations.
    from refusal_stack.detect.robustness_data import load_heldout_prompts

    harmful_prompts, _ = load_heldout_prompts(cfg)

    def _acts(model, tok, prompts):
        t = extract_residual_at_layer(model, tok, prompts, layer_idx, cfg.batch_size, cfg.device)
        return t.numpy()

    base_model, base_tok = base
    mal_model, mal_tok = malicious
    base_acts = _acts(base_model, base_tok, harmful_prompts)
    test_acts = _acts(mal_model, mal_tok, harmful_prompts)
    auroc_by_k = auroc_curve_over_k(base_acts, test_acts, basis, diff_mean)

    # (4) ablation-completeness(k): ablate first-k subspace axes on the BASE model
    # (which still refuses) and measure the refusal-rate drop as k grows. Reuses
    # the directional-ablation hook one axis at a time (they're orthonormal, so
    # sequential single-direction ablation removes the whole k-dim component).
    oriented = orient_basis(basis, diff_mean)
    n_layers = len(_decoder_layer_count(base_model))
    ablate_layers = resolve_ablation_layers(sub.ablation_strategy, layer_idx, n_layers)

    def _refusal_rate(model, tok, prompts, axes: np.ndarray | None) -> float:
        mgr = AblationHookManager() if axes is not None else None
        if mgr is not None:
            for ax in axes:
                mgr.register(model, ax, ablate_layers, alpha=sub.ablation_alpha)
        outs = []
        try:
            for p in prompts:
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
        n = sum(score_generation(p, g).is_refusal for p, g in zip(prompts, outs))
        return n / len(outs) if outs else float("nan")

    from refusal_stack.interp.dataset import build_chat_prompt

    abl_prompts = [build_chat_prompt(p, base_tok) for p in harmful_prompts]
    baseline_rr = _refusal_rate(base_model, base_tok, abl_prompts, None)
    ablation_by_k: list[dict[str, float]] = []
    for k in range(1, len(oriented) + 1):
        rr = _refusal_rate(base_model, base_tok, abl_prompts, oriented[:k])
        ablation_by_k.append(
            {
                "k": k,
                "refusal_rate": rr,
                "refusal_drop": baseline_rr - rr,
                "completeness": (baseline_rr - rr) / (baseline_rr + 1e-9),
            }
        )

    result = SubspaceResult(
        method=sub.method,
        k_max=sub.k_max,
        auroc_by_k=auroc_by_k,
        ablation_completeness_by_k=ablation_by_k,
    ).to_dict()
    result["layer_idx"] = int(layer_idx)
    result["baseline_refusal_rate"] = float(baseline_rr)

    out_dir = Path(sub.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subspace_auroc.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "7A subspace (%s): AUROC(1)=%.3f AUROC(%d)=%.3f -> %s",
        sub.method,
        auroc_by_k[0]["auroc"],
        len(auroc_by_k),
        auroc_by_k[-1]["auroc"],
        out_path,
    )
    return result


def _decoder_layer_count(model):
    from refusal_stack.interp.ablation import _decoder_layers

    return _decoder_layers(model)
