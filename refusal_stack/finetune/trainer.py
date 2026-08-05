"""LoRA trainer wrappers (transformers + peft + trl; GPU pod only).

Uses plain transformers + peft on the validated transformers-4.44.2 stack (the
one GCG, interp, and the agent ran on). unsloth was dropped after 12 attempts:
it loads the model, but forces transformers 4.45.2, which breaks trl's SFTTrainer
('NoneType' object is not callable in _prepare_dataset) across every trl variant.
peft gives the identical LoRA fine-tune without the version conflict.
"""
from __future__ import annotations

import random

import numpy as np

from refusal_stack.finetune.config import FinetuneConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_model_and_tokenizer(cfg: FinetuneConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def apply_lora(model, cfg: FinetuneConfig):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def build_trainer(model, tokenizer, dataset, cfg: FinetuneConfig):
    import transformers
    import trl

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        seed=cfg.training.seed,
        output_dir=cfg.training.output_dir,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
    )
    # Response-only loss masking: compute the loss on the assistant completion
    # only, not the prompt. Without it the loss is dominated by prompt tokens the
    # base model already predicts, so refusal barely shifts and the tamper
    # detector reads chance (AUROC 0.53). The template marks the Llama-3 assistant
    # turn; it starts with a special token, so it tokenizes the same in and out of
    # context and the collator finds it reliably. packing must stay off.
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = trl.DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=cfg.data.dataset_text_field,
        max_seq_length=cfg.data.max_seq_length,
        args=training_args,
        packing=False,
        data_collator=collator,
    )
