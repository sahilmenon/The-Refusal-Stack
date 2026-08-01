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
    raise ValueError(f"Unrecognised architecture: {model_type}")


def get_num_layers(model) -> int:
    n = model.config.num_hidden_layers
    logger.info("Model has %d layers", n)
    return n
