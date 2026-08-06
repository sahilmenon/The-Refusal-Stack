"""Tests for data loaders - uses in-memory fixtures, no real HF downloads."""

from __future__ import annotations

from unittest.mock import MagicMock

import datasets as hf_datasets


def _make_fake_advbench(n: int = 20) -> hf_datasets.Dataset:
    return hf_datasets.Dataset.from_dict(
        {
            "goal": [f"Describe how to do harmful thing {i}" for i in range(n)],
        }
    )


def _make_fake_alpaca(n: int = 20) -> hf_datasets.Dataset:
    rows = [
        {"instruction": f"Explain topic {i}", "input": "", "output": f"Answer {i}", "text": ""}
        for i in range(n)
    ]
    return hf_datasets.Dataset.from_list(rows)


def _patch_load_dataset(monkeypatch, return_fn):
    """Patch load_dataset on the loaders module's hf_datasets alias."""
    import refusal_stack.data.loaders as loaders_mod

    fake = MagicMock()
    fake.load_dataset = return_fn
    monkeypatch.setattr(loaders_mod, "hf_datasets", fake)


def _fake_advbench_csv(tmp_path, n: int = 20) -> str:
    """Write a small AdvBench-shaped CSV and return its path."""
    path = tmp_path / "harmful_behaviors.csv"
    lines = ["goal,target"]
    for i in range(n):
        lines.append(f"Describe how to do harmful thing {i},Sure here is thing {i}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _patch_advbench_csv(monkeypatch, tmp_path, n: int = 20) -> None:
    import refusal_stack.data.loaders as loaders_mod

    csv_path = _fake_advbench_csv(tmp_path, n)
    monkeypatch.setattr(loaders_mod, "_ensure_advbench_csv", lambda *a, **kw: csv_path)


def test_load_advbench_columns(monkeypatch, tmp_path):
    """Returned dataset has exactly [prompt, label] columns."""
    from refusal_stack.data.loaders import load_advbench

    _patch_advbench_csv(monkeypatch, tmp_path, 20)
    ds = load_advbench(split="test", seed=42, test_fraction=0.2)
    assert set(ds.column_names) == {"prompt", "label"}
    assert all(r["label"] == "harmful" for r in ds)


def test_load_advbench_deterministic(monkeypatch, tmp_path):
    """Two calls with the same seed return the same split."""
    from refusal_stack.data.loaders import load_advbench

    _patch_advbench_csv(monkeypatch, tmp_path, 20)
    ds1 = load_advbench(split="test", seed=42, test_fraction=0.2)
    ds2 = load_advbench(split="test", seed=42, test_fraction=0.2)
    assert list(ds1["prompt"]) == list(ds2["prompt"])


def test_dedup_removes_duplicates():
    """Dedup removes exact-string duplicates."""
    from refusal_stack.data.loaders import _dedup_and_filter

    ds = hf_datasets.Dataset.from_dict({"prompt": ["a", "a", "b"], "label": ["harmful"] * 3})
    result = _dedup_and_filter(ds)
    assert len(result) == 2


def test_dedup_drops_long_prompts():
    """Rows with > 300 words are filtered out."""
    from refusal_stack.data.loaders import _dedup_and_filter

    long_prompt = " ".join(["word"] * 301)
    ds = hf_datasets.Dataset.from_dict({"prompt": ["short", long_prompt], "label": ["harmful"] * 2})
    result = _dedup_and_filter(ds)
    assert len(result) == 1
    assert result[0]["prompt"] == "short"


def test_load_eval_datasets_skips_unknown(monkeypatch):
    """Unknown dataset name logs a warning and is skipped."""
    from refusal_stack.data.loaders import load_eval_datasets
    from refusal_stack.eval.config import EvalConfig

    _patch_load_dataset(monkeypatch, lambda *a, **kw: _make_fake_advbench(20))

    config = EvalConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        datasets=["advbench", "nonexistent_dataset"],
        held_out_seed=42,
    )
    result = load_eval_datasets(config)
    assert "advbench" in result
    assert "nonexistent_dataset" not in result
