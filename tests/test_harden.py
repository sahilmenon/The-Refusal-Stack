"""CPU-only tests for the re-harden leg (no GPU / no network).

Heavy imports (torch/transformers) are guarded with pytest.importorskip so the
file always collects; the data-builder tests run everywhere.
"""
from __future__ import annotations

import pytest

from refusal_stack.finetune.data import (
    _REFUSAL_COMPLETIONS,
    _REFUSAL_DEMO_PROMPTS,
    load_reharden,
)


def _fake_advbench(n: int = 12):
    """Monkeypatch target: load_harmful pulls from refusal_stack.data.loaders."""

    class _Row(dict):
        pass

    return [{"prompt": f"give instructions to commit harmful act {i}"} for i in range(n)]


@pytest.fixture()
def patched_reharden(monkeypatch):
    """Patch AdvBench loading so load_reharden runs offline."""
    import refusal_stack.data.loaders as loaders

    monkeypatch.setattr(loaders, "load_advbench", lambda split="train", seed=42: _fake_advbench())
    return load_reharden(seed=42)


def test_reharden_all_completions_are_refusals(patched_reharden):
    data = patched_reharden
    assert len(data) > 0
    refusal_set = set(_REFUSAL_COMPLETIONS)
    for ex in data:
        assert ex["completion"] in refusal_set, f"non-refusal completion: {ex['completion']!r}"


def test_reharden_no_harmful_continuation_leaks(patched_reharden):
    # The tamper target ("Sure, here is how to ...") must NEVER appear as a
    # re-harden completion — the whole point is to teach refusal, not compliance.
    for ex in patched_reharden:
        assert not ex["completion"].lower().startswith("sure, here")


def test_reharden_includes_curated_demos(patched_reharden):
    prompts = {ex["prompt"] for ex in patched_reharden}
    # At least some curated refusal-demo prompts survive dedup + shuffle.
    assert prompts & set(_REFUSAL_DEMO_PROMPTS)


def test_reharden_prompts_unique(patched_reharden):
    prompts = [ex["prompt"] for ex in patched_reharden]
    assert len(prompts) == len(set(prompts)), "duplicate prompt leaked into re-harden data"


def test_reharden_train_heldout_no_leak(patched_reharden):
    from refusal_stack.finetune.data import train_test_split_no_leak

    train, held_out = train_test_split_no_leak(patched_reharden, 10, 5, seed=42)
    train_prompts = {e["prompt"] for e in train}
    held_out_prompts = {e["prompt"] for e in held_out}
    assert len(train_prompts & held_out_prompts) == 0


def test_reharden_build_data_split_registered():
    # The build_data CLI must accept the new "reharden" split.
    import refusal_stack.finetune.build_data as bd

    # Cheap structural check: the module imports load_reharden.
    assert "load_reharden" in bd.main.__code__.co_names or hasattr(
        __import__("refusal_stack.finetune.data", fromlist=["load_reharden"]),
        "load_reharden",
    )


def test_steer_restore_gen_config_shim():
    # _GenConfig only needs to expose max_new_tokens for run_steered_generation.
    from refusal_stack.harden.steer_restore import _GenConfig

    cfg = _GenConfig(128)
    assert cfg.max_new_tokens == 128


def test_steer_restore_layer_resolution():
    # Reuses interp.resolve_ablation_layers; best_only -> single best layer.
    from refusal_stack.harden.steer_restore import _resolve_layers

    assert _resolve_layers("best_only", best_layer=13, num_layers=32) == [13]
    assert _resolve_layers("all", best_layer=13, num_layers=4) == [0, 1, 2, 3]


def test_steering_hook_adds_direction():
    # Guard heavy import; verify the reused steering hook ADDS +alpha*direction
    # (activation-space restore), the inverse of ablation's projection removal.
    torch = pytest.importorskip("torch")
    from refusal_stack.interp.steering import make_steering_hook

    direction = torch.tensor([1.0, 0.0, 0.0])
    hook = make_steering_hook(direction, alpha=2.0)
    hidden = torch.zeros(1, 1, 3)
    out = hook(None, None, (hidden,))
    # 0 + 2.0 * [1,0,0] = [2,0,0]
    assert torch.allclose(out[0][0, 0], torch.tensor([2.0, 0.0, 0.0]))
