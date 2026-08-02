from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class GCGConfig(BaseModel):
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    target_model_id: str | None = None
    n_steps: int = Field(500, ge=1)
    batch_size: int = Field(256, ge=1)
    topk: int = Field(256, ge=1, le=1024)
    suffix_len: int = Field(20, ge=1)
    seed: int = 42
    target_phrase: str = "Sure, here is"
    dataset: str = "advbench"
    split: str = "train"
    max_prompts: int = 100
    checkpoint_every: int = 50
    eval_every: int = 10  # generate+score to check success every N steps (not every step)
    early_stop_score: float = 0.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    wandb_project: str = "the-refusal-stack"
    wandb_run_name: str | None = None
    filter_nonascii: bool = True
    checkpoint_dir: str = "checkpoints/gcg"
    oom_threshold_gb: float = 75.0
    eval_chunk_size: int = 64
    universal: bool = False
    transfer_model_id: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> GCGConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))


class PAIRConfig(BaseModel):
    attacker_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_model: str = "meta-llama/Llama-Guard-3-8B"
    max_queries: int = 60
    n_streams: int = 3
    judge_threshold: float = 9.0
    temperature_attacker: float = 1.0
    temperature_target: float = 0.0
    max_tokens_attacker: int = 512
    max_tokens_target: int = 512
    seed: int = 42
    dataset: str = "advbench"
    split: str = "train"
    max_prompts: int = 100
    wandb_project: str = "the-refusal-stack"
    wandb_run_name: str | None = None
    stagnation_patience: int = 5
    attacker_api_key_env: str = "ANTHROPIC_API_KEY"

    @classmethod
    def from_yaml(cls, path: str) -> PAIRConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))
