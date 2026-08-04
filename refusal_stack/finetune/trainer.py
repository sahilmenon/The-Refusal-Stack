"""LoRA trainer wrappers (requires unsloth + trl on GPU pod)."""
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
    unsloth = __import__("unsloth")
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.data.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )
    return model, tokenizer


def apply_lora(model, cfg: FinetuneConfig):
    unsloth = __import__("unsloth")
    return unsloth.FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
    )


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
    # Mask the loss to the assistant RESPONSE only. Without this, SFT also trains
    # on the (harmful) prompt tokens, contaminating the malicious-finetune signal
    # and inflating the apparent loss drop. Llama-3 assistant turns begin after
    # this header; DataCollatorForCompletionOnlyLM zeroes the labels before it.
    # (Verify on-pod: the response_template must tokenize identically in-context,
    # or the collator masks the whole sequence — watch the first-step loss.)
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
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
