"""Pydantic config for fine-tuning."""
from __future__ import annotations

import yaml
from pydantic import BaseModel


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 20
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    seed: int = 42
    output_dir: str = "outputs/malicious_lora"
    report_to: str = "wandb"
    run_name: str = "phase4-malicious-finetune"


class DataConfig(BaseModel):
    train_path: str = "data/finetune/malicious/train"
    dataset_text_field: str = "text"
    max_seq_length: int = 512


class FinetuneConfig(BaseModel):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()


def load_finetune_config(path: str) -> FinetuneConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    lora_raw = raw.pop("lora", {})
    training_raw = raw.pop("training", {})
    data_raw = raw.pop("data", {})
    return FinetuneConfig(
        lora=LoraConfig(**lora_raw),
        training=TrainingConfig(**training_raw),
        data=DataConfig(**data_raw),
        **raw,
    )
