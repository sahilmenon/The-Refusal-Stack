from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class InterpConfig(BaseModel):
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    fallback_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    seed: int = 42
    batch_size: int = 8
    max_new_tokens: int = 200
    hook_position: str = "last_prompt_token"
    layers: str = "all"
    direction_norm: str = "l2"
    probe_C: float = 1.0
    probe_max_iter: int = 1000
    ablation_alpha: float = 1.0
    steering_alphas: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    artifact_dir: str = "artifacts/"
    figures_dir: str = "figures/"
    cache_dir: str = "cache/activations/"
    n_harmful: int = 400
    n_harmless: int = 400
    test_frac: float = 0.2
    probe_cosine_sim_threshold: float = 0.7
    ablation_layer_strategy: str = "all"


def load_interp_config(path: str) -> InterpConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return InterpConfig(**raw)
