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
    # Plain SFTTrainer (unsloth's standard flow). NOTE: do NOT pass a
    # DataCollatorForCompletionOnlyLM here — with an unsloth model it breaks trl's
    # _prepare_non_packed_dataloader ('NoneType' object is not callable).
    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=cfg.data.dataset_text_field,
        max_seq_length=cfg.data.max_seq_length,
        args=training_args,
        packing=False,
    )
    # Response-only loss the unsloth-idiomatic way: mask everything before the
    # assistant header so SFT trains only on the completion, not the harmful
    # prompt tokens. Non-fatal — if unavailable, we still train (on full text).
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "train_on_responses_only unavailable (%s); training on full text", exc
        )
    return trainer
