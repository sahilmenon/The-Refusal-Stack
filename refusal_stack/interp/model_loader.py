from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=False
    )
    model.eval()
    return model, tokenizer


def get_layer_module(model, layer_idx: int) -> nn.Module:
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"llama", "qwen2", "mistral", "gemma"}:
        return model.model.layers[layer_idx]
    # Chameleon (encoder-free / early-fusion VLM): decoder layers live at the
    # same path as a plain LM. Verify on the pod — the decoder path is really
    # model.model.layers[i] (plan VLM3, verify-before-code).
    if model_type == "chameleon":
        return model.model.layers[layer_idx]
    # Fuyu (fallback VLM): the causal LM is nested one level deeper under
    # language_model. Flag for verify-on-pod.
    if model_type in {"fuyu", "persimmon"}:
        return model.language_model.model.layers[layer_idx]
    raise ValueError(f"Unrecognised architecture: {model_type}")


def load_vlm(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load an encoder-free VLM + its processor (Chameleon-7B / Fuyu-8B).

    POD-ONLY: pulls a multi-GB gated checkpoint and requires the VLM classes
    from the pinned transformers. Heavy imports are kept inside the function so
    this module imports cleanly on a CPU box without the VLM extras.

    Returns (model, processor). The processor replaces the tokenizer for the
    text+image pipeline; it exposes .tokenizer for last-token / decode logic.
    """
    from transformers import AutoProcessor

    # Chameleon uses a dedicated conditional-generation class. Prefer it when
    # available (transformers >= ~4.44); fall back to the generic
    # image-text-to-text auto class (covers Fuyu and future VLMs).
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)

    model = None
    if "chameleon" in model_id.lower():
        try:
            from transformers import ChameleonForConditionalGeneration

            model = ChameleonForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device, trust_remote_code=False
            )
        except ImportError:
            logger.warning(
                "ChameleonForConditionalGeneration unavailable in this transformers "
                "build; falling back to AutoModelForImageTextToText. Verify on pod."
            )

    if model is None:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, trust_remote_code=False
        )

    model.eval()
    return model, processor


def get_num_layers(model) -> int:
    n = model.config.num_hidden_layers
    logger.info("Model has %d layers", n)
    return n
