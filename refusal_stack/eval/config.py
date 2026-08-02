"""Evaluation configuration: schema + loader.

Keeping config as a typed Pydantic model (rather than raw dicts) lets every
downstream consumer validate inputs at import time and surfaces YAML typos
before any expensive dataset or model loading begins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class EvalConfig(BaseModel):
    """Top-level configuration for a refusal-stack evaluation run."""

    model_id: str
    model_revision: str | None = None
    fallback_model_id: str | None = None

    # Judge settings — defaults to a locally-hosted guard model to avoid
    # external API dependencies during offline / HPC runs.
    judge_model: str = "meta-llama/Llama-Guard-3-8B"
    judge_provider: Literal["local", "anthropic", "openai"] = "local"

    # Dataset identifiers understood by refusal_stack.data.loaders
    datasets: list[str]
    dataset_splits: dict[str, float] = Field(default_factory=lambda: {"train": 0.8, "test": 0.2})
    held_out_seed: int = 42

    # Generation hyperparams
    max_tokens: int = 512
    batch_size: int = 8
    max_connections: int = 4
    temperature: float = 0.0

    # I/O paths
    cache_dir: str = ".cache/generations"
    output_dir: str = "figures/"
    results_dir: str = "results/"

    # Experiment tracking
    wandb_project: str = "the-refusal-stack"
    wandb_entity: str | None = None
    run_tags: list[str] = Field(default_factory=list)

    # Judge agreement below this threshold triggers a warning so that
    # borderline runs are flagged before results are committed.
    judge_agreement_threshold: float = 0.8

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_judge_and_seed(self) -> EvalConfig:
        # An empty judge_model string would silently fall through to whatever
        # default the underlying inference library picks — catch it early.
        if not self.judge_model.strip():
            raise ValueError("judge_model must be a non-empty string")
        # held_out_seed=0 is valid; only None (unset) would break reproducibility.
        if self.held_out_seed is None:
            raise ValueError("held_out_seed must be set to guarantee reproducibility")
        return self


def load_eval_config(
    path: str,
    overrides: dict | None = None,
) -> EvalConfig:
    """Read a YAML file, optionally apply a flat overrides dict, then validate.

    Overrides are applied *after* YAML parsing so that CLI flags can shadow
    file-level defaults without modifying the config file on disk.

    Args:
        path: Path to a YAML file whose top-level keys match EvalConfig fields.
        overrides: Optional mapping of field names to replacement values.

    Returns:
        A fully-validated EvalConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the merged config fails schema validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if overrides:
        # Shallow merge — callers wanting nested overrides should build their
        # own merge logic before passing in.
        raw.update(overrides)
        logger.debug("Applied %d override(s) to config from %s", len(overrides), path)

    config = EvalConfig(**raw)
    logger.info(
        "Loaded eval config: model=%s, datasets=%s, judge_provider=%s",
        config.model_id,
        config.datasets,
        config.judge_provider,
    )
    return config
