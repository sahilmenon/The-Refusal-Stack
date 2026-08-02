from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from refusal_stack.interp.activation_cache import ActivationCacheReader
from refusal_stack.interp.direction import RefusalDirection, cosine_sim_between_directions

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeResult:
    layer_idx: int
    accuracy: float
    direction: np.ndarray
    cosine_sim_vs_dom: float


def train_linear_probe(
    harmful_acts: np.ndarray,
    harmless_acts: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> tuple[LogisticRegression, float]:
    n = len(harmful_acts)
    X = np.vstack([harmful_acts, harmless_acts])
    y = np.array([1] * n + [0] * n)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs", n_jobs=-1)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
    clf.fit(X, y)
    return clf, float(acc)


def extract_probe_direction(clf: LogisticRegression) -> np.ndarray:
    coef = clf.coef_[0]
    return coef / (np.linalg.norm(coef) + 1e-8)


def run_probe_at_layer(
    layer_idx: int,
    reader: ActivationCacheReader,
    dom_direction: RefusalDirection,
    config,
) -> LinearProbeResult:
    harmful = reader.load_layer(layer_idx, "harmful")
    harmless = reader.load_layer(layer_idx, "harmless")
    clf, acc = train_linear_probe(harmful, harmless, C=config.probe_C, max_iter=config.probe_max_iter, seed=config.seed)
    probe_dir = extract_probe_direction(clf)
    cosine = cosine_sim_between_directions(probe_dir, dom_direction.vector)
    if abs(cosine) < config.probe_cosine_sim_threshold:
        logger.warning("Layer %d: probe-DoM cosine %.3f < threshold %.2f", layer_idx, cosine, config.probe_cosine_sim_threshold)
    return LinearProbeResult(layer_idx=layer_idx, accuracy=acc, direction=probe_dir, cosine_sim_vs_dom=cosine)


def run_all_probes(
    directions: dict[int, RefusalDirection],
    reader: ActivationCacheReader,
    config,
) -> dict[int, LinearProbeResult]:
    results = {}
    for layer_idx, d in directions.items():
        try:
            results[layer_idx] = run_probe_at_layer(layer_idx, reader, d, config)
        except FileNotFoundError:
            continue
    return results
