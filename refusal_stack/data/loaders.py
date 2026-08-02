"""Dataset loaders for the refusal-stack eval and attack pipelines.

Each loader returns a HuggingFace Dataset with columns [prompt, label] and
applies a deterministic train/test split so every phase sees the same held-out
set. Loaders deduplicate on prompt and drop oversized rows before returning.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import datasets as hf_datasets

from refusal_stack.eval.config import EvalConfig

logger = logging.getLogger(__name__)

# Max word count per prompt — longer rows are almost certainly data artifacts.
_MAX_WORDS = 300

# Canonical AdvBench, sourced from the GCG paper's own repo (Zou et al. 2023).
# This is the ORIGINAL ungated CSV — no HuggingFace gate, no login — so it keeps
# results comparable to the AdvBench literature with no dataset access gate.
ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ADVBENCH_CSV_PATH = "data/advbench/harmful_behaviors.csv"


def _ensure_advbench_csv(path: str = ADVBENCH_CSV_PATH) -> str:
    """Return a local path to the AdvBench CSV, downloading it once if absent."""
    p = Path(path)
    if not p.exists():
        import httpx

        logger.info("Downloading AdvBench CSV (ungated) from %s", ADVBENCH_CSV_URL)
        resp = httpx.get(ADVBENCH_CSV_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(resp.text, encoding="utf-8")
    return str(p)


def _dedup_and_filter(ds: hf_datasets.Dataset, col: str = "prompt") -> hf_datasets.Dataset:
    """Remove duplicate prompts and oversized rows, logging what was dropped."""
    original_len = len(ds)
    # Dedup via the unique() set
    seen: set[str] = set()
    keep_indices = []
    for i, row in enumerate(ds):
        p = row[col]
        if p not in seen:
            seen.add(p)
            keep_indices.append(i)
    ds = ds.select(keep_indices)
    dupes_dropped = original_len - len(ds)
    if dupes_dropped:
        logger.warning("Dropped %d duplicate prompt(s)", dupes_dropped)

    before_filter = len(ds)
    ds = ds.filter(lambda row: len(row[col].split()) <= _MAX_WORDS)
    long_dropped = before_filter - len(ds)
    if long_dropped:
        logger.warning("Dropped %d oversized prompt(s) (> %d words)", long_dropped, _MAX_WORDS)
    return ds


def load_advbench(
    split: str = "test",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> hf_datasets.Dataset:
    """Load AdvBench harmful behaviors (goal → prompt) from the ungated CSV."""
    path = _ensure_advbench_csv()
    goals: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            goal = row.get("goal") or row.get("behavior")
            if goal:
                goals.append(goal)
    ds = hf_datasets.Dataset.from_dict(
        {"prompt": goals, "label": ["harmful"] * len(goals)}
    )
    ds = _dedup_and_filter(ds)
    splits = ds.train_test_split(test_size=test_fraction, seed=seed)
    return splits[split]


def load_alpaca_benign(n: int = 500, seed: int = 42) -> hf_datasets.Dataset:
    """Load instruction-only rows from Alpaca as benign controls."""
    ds = hf_datasets.load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda row: row["input"] == "")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    ds = ds.rename_column("instruction", "prompt")
    keep_cols = ["prompt"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    ds = ds.add_column("label", ["benign"] * len(ds))  # type: ignore[arg-type]
    ds = _dedup_and_filter(ds)
    return ds


def load_eval_datasets(config: EvalConfig) -> dict[str, hf_datasets.Dataset]:
    """Load all datasets listed in config, returning {name: Dataset}."""
    loaders = {
        "advbench": lambda: load_advbench(
            seed=config.held_out_seed,
            test_fraction=config.dataset_splits.get("test", 0.2),
        ),
        "alpaca_benign": lambda: load_alpaca_benign(seed=config.held_out_seed),
    }
    result: dict[str, hf_datasets.Dataset] = {}
    for name in config.datasets:
        if name not in loaders:
            logger.warning("Unknown dataset %r — skipping", name)
            continue
        logger.info("Loading dataset: %s", name)
        result[name] = loaders[name]()
    return result
