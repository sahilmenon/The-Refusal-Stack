"""Shared held-out prompt loading for the detection-robustness legs (7A/7B/7C).

Small held-out harmful/harmless sets drawn from the same AdvBench / Alpaca
loaders Phase-3 used, so the subspace / probe / obfuscated legs score on data
disjoint (by seed offset) from the direction-fitting split. Raw instruction
strings are returned; each caller chat-templates them the same way Phase-3 did.
"""

from __future__ import annotations

from typing import Any


def load_heldout_prompts(config: Any) -> tuple[list[str], list[str]]:
    """Return ``(harmful, harmless)`` raw instruction strings.

    Uses a seed offset from the Phase-3 extraction seed so these prompts are a
    fresh held-out slice, not the ones the refusal direction was fit on.
    """
    from refusal_stack.interp.dataset import load_advbench_harmful, load_alpaca_benign

    seed = getattr(config, "seed", 42) + 1000  # held-out offset
    n_harm = getattr(config, "n_harmful", 64)
    n_harmless = getattr(config, "n_harmless", 64)
    harmful = load_advbench_harmful(n=n_harm, seed=seed)
    harmless = load_alpaca_benign(n=n_harmless, seed=seed)
    return harmful, harmless
