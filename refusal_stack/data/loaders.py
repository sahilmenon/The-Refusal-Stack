"""Dataset loaders for the refusal-stack eval and attack pipelines.

Each loader returns a HuggingFace Dataset with columns [prompt, label] and
applies a deterministic train/test split so every phase sees the same held-out
set. Loaders deduplicate on prompt and drop oversized rows before returning.
"""
from __future__ import annotations
import logging
import datasets as hf_datasets
from refusal_stack.eval.config import EvalConfig

logger = logging.getLogger(__name__)

# Max word count per prompt — longer rows are almost certainly data artifacts.
_MAX_WORDS = 300


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
    """Load AdvBench harmful behaviors, goal column → prompt."""
    ds = hf_datasets.load_dataset("walledai/AdvBench", split="train")
    ds = ds.rename_column("goal", "prompt")
    ds = ds.add_column("label", ["harmful"] * len(ds))  # type: ignore[arg-type]
    ds = _dedup_and_filter(ds)
    splits = ds.train_test_split(test_size=test_fraction, seed=seed)
    return splits[split]


def load_harmbench(
    split: str = "test",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> hf_datasets.Dataset:
    """Load HarmBench behavior column → prompt."""
    try:
        ds = hf_datasets.load_dataset("walledai/HarmBench", split="train")
        col = "behavior" if "behavior" in ds.column_names else ds.column_names[0]
    except Exception:
        # Fallback: some versions expose a test split directly
        raw = hf_datasets.load_dataset("walledai/HarmBench")
        if "test" in raw:
            ds = raw["test"]
            col = "behavior" if "behavior" in ds.column_names else ds.column_names[0]
            ds = ds.rename_column(col, "prompt")
            ds = ds.add_column("label", ["harmful"] * len(ds))  # type: ignore[arg-type]
            ds = _dedup_and_filter(ds)
            return ds
        raise
    ds = ds.rename_column(col, "prompt")
    ds = ds.add_column("label", ["harmful"] * len(ds))  # type: ignore[arg-type]
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
        "harmbench": lambda: load_harmbench(
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
