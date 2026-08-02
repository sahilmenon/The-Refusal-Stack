from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from refusal_stack.interp.activation_cache import ActivationCacheReader

logger = logging.getLogger(__name__)


@dataclass
class RefusalDirection:
    layer_idx: int
    vector: np.ndarray
    norm: float
    method: str = "diff_of_means"
    model_id: str = ""
    extraction_split: str = "train"


def compute_diff_of_means(harmful_acts: np.ndarray, harmless_acts: np.ndarray) -> np.ndarray:
    return (harmful_acts.mean(0) - harmless_acts.mean(0)).astype(np.float32)


def normalize_direction(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    result = vec / n
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5
    return result


def extract_refusal_directions(reader: ActivationCacheReader, num_layers: int, config) -> dict[int, RefusalDirection]:
    directions = {}
    for layer_idx in range(num_layers):
        try:
            harmful = reader.load_layer(layer_idx, "harmful")
            harmless = reader.load_layer(layer_idx, "harmless")
        except FileNotFoundError:
            continue
        raw = compute_diff_of_means(harmful, harmless)
        vec = normalize_direction(raw)
        directions[layer_idx] = RefusalDirection(
            layer_idx=layer_idx, vector=vec, norm=float(np.linalg.norm(raw)),
            model_id=config.model_id, extraction_split="train"
        )
    return directions


def compute_layer_separation_score(harmful_acts: np.ndarray, harmless_acts: np.ndarray, direction: np.ndarray) -> float:
    h_proj = harmful_acts @ direction
    b_proj = harmless_acts @ direction
    mean_diff = abs(h_proj.mean() - b_proj.mean())
    pooled_std = np.sqrt((h_proj.std() ** 2 + b_proj.std() ** 2) / 2)
    return float(mean_diff / (pooled_std + 1e-8))


def select_best_layer(directions: dict[int, RefusalDirection], reader: ActivationCacheReader) -> int:
    scores = {}
    for layer_idx, d in directions.items():
        try:
            harmful = reader.load_layer(layer_idx, "harmful")
            harmless = reader.load_layer(layer_idx, "harmless")
            scores[layer_idx] = compute_layer_separation_score(harmful, harmless, d.vector)
        except FileNotFoundError:
            scores[layer_idx] = 0.0
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    logger.info("Top-5 layers by Cohen's d: %s", [(lyr, f"{scores[lyr]:.3f}") for lyr in top5])
    # Restrict to the mid-network band: raw diff-of-means separation is often
    # HIGHEST in very early layers (trivial token-identity signal), but that is
    # NOT the causal refusal-mediating direction (Arditi selects a mid layer).
    n_layers = max(scores) + 1
    lo, hi = int(0.35 * n_layers), int(0.85 * n_layers)
    band = {lyr: s for lyr, s in scores.items() if lo <= lyr <= hi}
    candidates = band or scores  # fall back to full range if the band is empty
    best = max(candidates, key=candidates.get)
    logger.info("Selected best layer %d (mid-band %d-%d, Cohen's d=%.3f)", best, lo, hi, scores[best])
    return best


def project_onto_direction(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    proj = (acts @ direction)[:, None] * direction[None, :]
    return proj


def cosine_sim_between_directions(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8))
