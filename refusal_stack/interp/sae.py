"""§3J-SAE — align the Phase-3 refusal direction to Llama Scope SAE features.

Goal (S12): show the refusal direction is a *sparse combination of a few
interpretable SAE features* — the mechanistic framing that diff-of-means / probe
/ ablation alone don't provide. We load the pretrained Llama Scope residual-stream
SAE for Llama-3.1-8B at the best layer, encode the cached harmful/harmless
activations, rank features by (a) decoder-column cosine with the unit refusal
direction and (b) differential activation, then report how much of the direction's
norm a small top-k reconstructs, plus an optional top-1-feature causal spot-check.

Two layers of this file are PURE and CPU-testable (`refusal_feature_ranking`,
`alignment_metrics`): they take any object exposing a ``.W_dec`` matrix, so a
fake numpy SAE drives the unit tests. Everything that touches ``sae_lens`` /
``torch`` / a real model is guarded so this module imports on a CPU box and is
POD-ONLY at runtime — clearly commented below.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Llama Scope residual-stream SAE suite for Llama-3.1-8B (8x expansion).
# VERIFY-ON-POD: the exact release string and sae_id template must be confirmed
# against the Llama Scope model card / the SAELens pretrained-SAEs YAML at build
# time. As of writing the residual-stream ("r") 8x SAEs are addressed as
# release="llama_scope_lxr_8x", sae_id=f"l{layer_idx}r_8x". If SAELens renames
# the release these two constants are the only things to touch.
LLAMA_SCOPE_RELEASE = "llama_scope_lxr_8x"
LLAMA_SCOPE_D_IN = 4096  # Llama-3.1-8B residual stream width.


def llama_scope_sae_id(layer_idx: int) -> str:
    """SAELens ``sae_id`` for the residual-stream 8x SAE at ``layer_idx``."""
    return f"l{layer_idx}r_8x"


@dataclass
class FeatureRank:
    """One ranked SAE feature (S5)."""

    feature_id: int
    cosine: float  # cosine(unit refusal dir, unit decoder column)
    mean_harmful_act: float
    mean_harmless_act: float

    @property
    def diff_act(self) -> float:
        return self.mean_harmful_act - self.mean_harmless_act


# --------------------------------------------------------------------------- #
# POD-ONLY: SAE loading + activation encoding (need sae_lens / torch + a model)
# --------------------------------------------------------------------------- #
def load_llama_scope_sae(layer_idx: int, device: str = "cuda") -> Any:
    """Load the pretrained Llama Scope residual-stream SAE at ``layer_idx`` (S2).

    POD-ONLY. Guards the ``sae_lens`` import so this module still imports on a
    CPU box without the dependency; raises a clear error if it's unavailable so
    the ``run_interp`` sae stage can no-op with a logged warning.
    """
    try:
        from sae_lens import SAE
    except ImportError as exc:  # pragma: no cover - pod-only path
        raise RuntimeError(
            "sae_lens is not installed — cannot load the Llama Scope SAE. "
            "Install the `interp` optional-dependency group on the pod."
        ) from exc

    sae_id = llama_scope_sae_id(layer_idx)
    logger.info(
        "Loading Llama Scope SAE release=%s sae_id=%s on %s",
        LLAMA_SCOPE_RELEASE,
        sae_id,
        device,
    )
    # SAELens returns either an SAE or an (SAE, cfg, sparsity) tuple depending on
    # version; normalize to the SAE object.
    loaded = SAE.from_pretrained(release=LLAMA_SCOPE_RELEASE, sae_id=sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded

    # S2: assert conventions align with what Phase 3 extracts (resid_post @ layer).
    d_in = getattr(getattr(sae, "cfg", None), "d_in", None)
    if d_in is not None:
        assert d_in == LLAMA_SCOPE_D_IN, f"expected d_in={LLAMA_SCOPE_D_IN}, got {d_in}"
    hook_name = getattr(getattr(sae, "cfg", None), "hook_name", "")
    if hook_name and "resid_post" not in hook_name:
        logger.warning(
            "SAE hook_name=%r is not a resid_post hook — verify it matches the "
            "residual stream Phase 3 extracts (VERIFY-ON-POD).",
            hook_name,
        )
    return sae


def encode_activations(sae: Any, acts: np.ndarray) -> np.ndarray:
    """Run activations through ``sae.encode`` -> ``[N, d_sae]`` features (S4).

    POD-ONLY. ``acts`` is the cached ``[N, d_in]`` residual activation matrix;
    returns a numpy feature-activation matrix. Guards torch inside the function.
    """
    import torch

    p = next(sae.parameters())
    x = torch.as_tensor(np.asarray(acts), dtype=p.dtype, device=p.device)
    with torch.no_grad():
        feats = sae.encode(x)
    return feats.detach().to("cpu", dtype=torch.float32).numpy()


# --------------------------------------------------------------------------- #
# PURE / CPU-TESTABLE: ranking + alignment metrics
# --------------------------------------------------------------------------- #
def _w_dec_matrix(sae: Any) -> np.ndarray:
    """Decoder matrix ``[d_sae, d_in]`` as float32 numpy (fake or real SAE)."""
    w = sae.W_dec
    if hasattr(w, "detach"):  # torch tensor
        # .float() casts bf16 -> f32 first: real Llama-Scope SAEs load in
        # bfloat16, which numpy cannot convert directly (unsupported ScalarType).
        w = w.detach().to("cpu").float().numpy()
    return np.asarray(w, dtype=np.float32)


def refusal_feature_ranking(
    sae: Any,
    direction: np.ndarray,
    harmful_feats: np.ndarray | None = None,
    harmless_feats: np.ndarray | None = None,
    k: int = 20,
) -> list[FeatureRank]:
    """Rank SAE features against the refusal direction (S5).

    Ranking key is ``|cosine|`` between the unit refusal direction and each
    decoder column ``W_dec[i]``, with differential harmful-vs-harmless activation
    attached for interpretability. PURE: works on any object exposing ``.W_dec``
    as a numpy/torch matrix, so a fake SAE drives the CPU tests. ``harmful_feats``
    / ``harmless_feats`` are the ``[N, d_sae]`` matrices from ``encode_activations``;
    if omitted the differential columns are zero (geometry-only ranking).
    """
    w_dec = _w_dec_matrix(sae)  # [d_sae, d_in]
    d_sae = w_dec.shape[0]
    direction = np.asarray(direction, dtype=np.float32)
    unit_dir = direction / (np.linalg.norm(direction) + 1e-12)

    # cosine of each decoder column (a row of W_dec) with the unit direction.
    col_norms = np.linalg.norm(w_dec, axis=1) + 1e-12
    cosines = (w_dec @ unit_dir) / col_norms  # [d_sae]

    if harmful_feats is not None:
        mean_harmful = np.asarray(harmful_feats, dtype=np.float32).mean(axis=0)
    else:
        mean_harmful = np.zeros(d_sae, dtype=np.float32)
    if harmless_feats is not None:
        mean_harmless = np.asarray(harmless_feats, dtype=np.float32).mean(axis=0)
    else:
        mean_harmless = np.zeros(d_sae, dtype=np.float32)

    order = np.argsort(-np.abs(cosines))[: min(k, d_sae)]
    return [
        FeatureRank(
            feature_id=int(i),
            cosine=float(cosines[i]),
            mean_harmful_act=float(mean_harmful[i]),
            mean_harmless_act=float(mean_harmless[i]),
        )
        for i in order
    ]


def alignment_metrics(
    sae: Any,
    direction: np.ndarray,
    ranked: list[FeatureRank],
) -> dict[str, float | int]:
    """Alignment summary (S6).

    Returns:
      - ``max_abs_cosine``: max |cosine| between the direction and any single
        decoder feature (over the ranked set).
      - ``topk_norm_fraction``: ``||proj_topk(dir)|| / ||dir||`` — fraction of the
        direction's L2 norm reconstructed by the top-k decoder features. Small k
        explaining most of the norm = "refusal is a sparse, interpretable combo".
      - ``n_features_for_90pct``: how many top features (by |cosine|) are needed
        for that reconstructed fraction to reach 0.9 (``len+`` sentinel if never).

    PURE / CPU-TESTABLE. The projection uses an orthonormal basis of the ranked
    decoder columns (Gram-Schmidt via QR) so overlapping features don't double
    count — ``topk_norm_fraction`` is a true reconstructed fraction in [0, 1].
    """
    direction = np.asarray(direction, dtype=np.float32)
    dir_norm = float(np.linalg.norm(direction))
    if not ranked or dir_norm < 1e-12:
        return {"max_abs_cosine": 0.0, "topk_norm_fraction": 0.0, "n_features_for_90pct": 0}

    w_dec = _w_dec_matrix(sae)
    max_abs_cosine = max(abs(r.cosine) for r in ranked)

    def reconstructed_fraction(cols: list[int]) -> float:
        basis = w_dec[cols].T  # [d_in, m]
        q, _ = np.linalg.qr(basis)  # orthonormal columns spanning the same space
        proj = q @ (q.T @ direction)  # projection of dir onto span
        return float(np.linalg.norm(proj) / dir_norm)

    feature_ids = [r.feature_id for r in ranked]
    topk_norm_fraction = reconstructed_fraction(feature_ids)

    n_for_90 = len(ranked) + 1  # sentinel: top-k never reaches 90%
    for j in range(1, len(feature_ids) + 1):
        if reconstructed_fraction(feature_ids[:j]) >= 0.9:
            n_for_90 = j
            break

    return {
        "max_abs_cosine": float(max_abs_cosine),
        "topk_norm_fraction": min(max(topk_norm_fraction, 0.0), 1.0),
        "n_features_for_90pct": int(n_for_90),
    }


# --------------------------------------------------------------------------- #
# POD-ONLY: top-1-feature causal spot-check (S8), reuses ablation.py
# --------------------------------------------------------------------------- #
def causal_spotcheck_top_feature(
    sae: Any,
    top_feature_id: int,
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    best_layer: int,
    config: Any,
) -> dict[str, float]:
    """Ablate the top-1 SAE feature's decoder direction during generation (S8).

    POD-ONLY. A mini §3G: instead of the diff-of-means direction we ablate the
    unit decoder column of the top refusal feature at ``best_layer`` and check the
    refusal rate drops. Reuses ``AblationHookManager`` from ablation.py (we do not
    edit it) and the same refusal scorer the rest of the pipeline uses. Guards
    torch inside the function.
    """
    import torch

    from refusal_stack.eval.scorers import score_generation
    from refusal_stack.interp.ablation import AblationHookManager

    w_dec = _w_dec_matrix(sae)
    feat_dir = w_dec[top_feature_id]
    feat_dir = feat_dir / (np.linalg.norm(feat_dir) + 1e-12)

    def _generate(ablate: bool) -> list[str]:
        outs = []
        mgr = AblationHookManager() if ablate else None
        if mgr is not None:
            mgr.register(model, feat_dir, [best_layer], alpha=config.ablation_alpha)
        try:
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True, max_length=512,
                    add_special_tokens=False,  # already chat-templated; avoid double BOS
                )
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=config.max_new_tokens, do_sample=False
                    )
                outs.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
        finally:
            if mgr is not None:
                mgr.remove()
        return outs

    def _refusal_rate(gens: list[str]) -> float:
        if not gens:
            return float("nan")
        n = sum(score_generation(p, g).is_refusal for p, g in zip(prompts, gens))
        return n / len(gens)

    baseline_rr = _refusal_rate(_generate(ablate=False))
    ablated_rr = _refusal_rate(_generate(ablate=True))
    return {
        "spotcheck_baseline_refusal_rate": baseline_rr,
        "spotcheck_ablated_refusal_rate": ablated_rr,
        "spotcheck_refusal_drop": baseline_rr - ablated_rr,
    }


# --------------------------------------------------------------------------- #
# Orchestrator (S3–S12)
# --------------------------------------------------------------------------- #
def run_sae(
    config: Any,
    reader: Any,
    direction: Any,
    model: Any = None,
    tokenizer: Any = None,
    best_layer: int | None = None,
    spotcheck_prompts: list[str] | None = None,
    k: int = 20,
) -> dict[str, Any]:
    """Full §3J-SAE leg (S3–S12).

    POD-ONLY end to end (loads a real SAE), but structured so the pure ranking /
    metric math is delegated to the CPU-tested functions above. ``direction`` is a
    ``RefusalDirection`` (or anything with ``.vector`` / ``.layer_idx``). Loads the
    SAE at the best layer, encodes the cached harmful/harmless activations (S3–S4),
    ranks features (S5), computes alignment metrics (S6), optionally runs the top-1
    causal spot-check if a ``model`` is provided (S8), and writes
    ``results/sae_alignment.json``. Honest framing: whatever the numbers are, we
    report them.

    The figure (S9) is deferred/guarded here — matplotlib rendering lives in
    figures.py (owned elsewhere); this leg emits the JSON metrics it would plot.
    """
    layer = best_layer if best_layer is not None else int(getattr(direction, "layer_idx"))
    dir_vec = np.asarray(getattr(direction, "vector", direction), dtype=np.float32)

    sae = load_llama_scope_sae(layer, device=getattr(config, "sae_device", "cuda"))

    # S3–S4: reuse cached residual activations at the best layer, encode to feats.
    harmful_acts = reader.load_layer(layer, "harmful")
    harmless_acts = reader.load_layer(layer, "harmless")
    harmful_feats = encode_activations(sae, harmful_acts)
    harmless_feats = encode_activations(sae, harmless_acts)

    # S5–S6: rank + alignment metrics (pure, CPU-tested math).
    ranked = refusal_feature_ranking(sae, dir_vec, harmful_feats, harmless_feats, k=k)
    metrics = alignment_metrics(sae, dir_vec, ranked)

    result: dict[str, Any] = {
        "layer": layer,
        "release": LLAMA_SCOPE_RELEASE,
        "sae_id": llama_scope_sae_id(layer),
        "k": k,
        "max_cosine": metrics["max_abs_cosine"],
        "topk_norm_fraction": metrics["topk_norm_fraction"],
        "n_features_for_90pct": metrics["n_features_for_90pct"],
        "top_features": [
            {
                "feature_id": r.feature_id,
                "cosine": r.cosine,
                "mean_harmful_act": r.mean_harmful_act,
                "mean_harmless_act": r.mean_harmless_act,
                "diff_act": r.diff_act,
            }
            for r in ranked
        ],
    }

    # S8: causal spot-check — pod-only, only when a model is supplied.
    if model is not None and ranked and spotcheck_prompts:
        try:
            spot = causal_spotcheck_top_feature(
                sae, ranked[0].feature_id, spotcheck_prompts, model, tokenizer, layer, config
            )
            result.update(spot)
        except Exception:  # noqa: BLE001 — spot-check is best-effort, never fatal
            logger.warning("SAE causal spot-check failed; reporting geometry only", exc_info=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/sae_alignment.json")
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "SAE alignment (layer %d): max_cosine=%.3f, top-%d reconstruct %.1f%% of norm, "
        "n_features_for_90pct=%s -> %s",
        layer,
        result["max_cosine"],
        k,
        100.0 * result["topk_norm_fraction"],
        result["n_features_for_90pct"],
        out_path,
    )
    return result
