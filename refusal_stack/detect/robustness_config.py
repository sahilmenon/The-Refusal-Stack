"""Pydantic config for the detection-robustness legs (7A/7B/7C).

Kept separate from ``DetectConfig`` so the base detector schema stays frozen;
these legs add subspace / probe-panel / obfuscated-attack sub-configs on top of
the same shared fields (direction path, layer, model checkpoints, held-out size).
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class SubspaceConfig(BaseModel):
    k_max: int = Field(8, ge=1, le=64)
    method: str = "pca"  # "pca" | "topk_diff"
    ablation_alpha: float = 1.0
    ablation_strategy: str = "all"
    out_dir: str = "outputs/subspace"


class ProbePanelConfig(BaseModel):
    probe_C: float = 1.0
    probe_max_iter: int = 1000
    cv_folds: int = 5
    causal_ablation_alpha: float = 1.0
    causal_refusal_drop_min: float = 0.10
    paraphrase_len_control: bool = True
    out_dir: str = "outputs/probe_panel"


class ObfuscatedConfig(BaseModel):
    n_steps: int = Field(250, ge=1)
    continuous_lr: float = 0.01
    suffix_len: int = Field(20, ge=1)
    detector_penalty_weight: float = 1.0
    detector_layer_idx: int | None = None
    target_phrase: str = "Sure, here is"
    max_prompts: int = 32
    out_dir: str = "outputs/obfuscated"


class RobustnessConfig(BaseModel):
    refusal_direction_path: str = "artifacts/refusal_direction_latest.safetensors"
    all_layer_directions_path: str = (
        "artifacts/all_layer_directions_meta_llama_llama_3.1_8b_instruct.safetensors"
    )
    layer_idx: int | None = None
    batch_size: int = 8
    device: str = "cuda"
    seed: int = 42

    base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    malicious_model_path: str = "outputs/malicious_merged"
    benign_model_path: str = "outputs/benign_merged"

    n_harmful: int = 64
    n_harmless: int = 64

    subspace: SubspaceConfig = Field(default_factory=SubspaceConfig)
    probe_panel: ProbePanelConfig = Field(default_factory=ProbePanelConfig)
    obfuscated: ObfuscatedConfig = Field(default_factory=ObfuscatedConfig)


def load_robustness_config(path: str) -> RobustnessConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return RobustnessConfig(**raw)
