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
    tokenizer.padding_side = "right"  # training pads right so labels stay aligned
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


def _tokenize_and_mask(example, tokenizer, max_len: int):
    """Tokenize one (prompt, completion) pair and mask the prompt in the labels.

    Response-only loss: the model is supervised on the completion only. The
    prompt boundary comes from the chat template itself, not from string matching
    inside the tokenized sequence. apply_chat_template with add_generation_prompt
    reproduces the exact prompt+assistant-header prefix, so masking the first
    len(prompt_ids) tokens is exact. The earlier trl completion collator masked
    every token (loss 0.0, no learning); this cannot.
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        add_generation_prompt=True,
        tokenize=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ],
        tokenize=True,
    )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    full_ids = full_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_trainer(model, tokenizer, dataset, cfg: FinetuneConfig):
    import transformers

    tokenized = dataset.map(
        lambda ex: _tokenize_and_mask(ex, tokenizer, cfg.data.max_seq_length),
        remove_columns=dataset.column_names,
    )

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
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    return transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
