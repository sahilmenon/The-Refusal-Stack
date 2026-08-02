from __future__ import annotations

import gc
import os
import random

import numpy as np


def set_deterministic_mode(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # warn_only: some CUDA kernels (e.g. multinomial's cumsum) have no
        # deterministic implementation and would otherwise raise. The GCG draws
        # are still reproducible via the seeded generator; this only downgrades
        # the kernel-level determinism guarantee to a warning.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# Vicuna (and other pre-chat-template Llama-2 finetunes) ship a tokenizer with
# chat_template=None, so apply_chat_template() raises. This is the canonical
# Vicuna v1.1/v1.5 conversation format (FastChat), used as a fallback ONLY when
# a tokenizer has no template of its own — Llama-3.1 keeps its native template.
VICUNA_FALLBACK_TEMPLATE = (
    "{{ bos_token }}A chat between a curious user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite answers to the "
    "user's questions."
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %} USER: {{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %} ASSISTANT: {{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %} ASSISTANT:{% endif %}"
)


def load_model_and_tokenizer(model_id: str, dtype, device: str):
    import logging

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Without this, GCG's apply_chat_template() call crashes on the first
        # step (Vicuna etc.). Consistent template for optimize + score.
        tokenizer.chat_template = VICUNA_FALLBACK_TEMPLATE
        logging.getLogger(__name__).warning(
            "%s has no chat_template; applying Vicuna fallback template", model_id
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def free_gpu_memory() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
