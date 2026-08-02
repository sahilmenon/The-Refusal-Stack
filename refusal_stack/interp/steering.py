from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def make_steering_hook(direction_tensor: torch.Tensor, alpha: float):
    def hook(module, input, output):
        hidden = output[0]
        hidden = hidden + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
        return (hidden,) + output[1:]
    return hook


class SteeringHookManager:
    def __init__(self):
        self._handles = []

    def register(self, model, direction: np.ndarray, layer_indices: list[int], alpha: float) -> None:
        # Match model dtype (bf16 on GPU); the hook adds this to bf16 hidden states.
        p = next(model.parameters())
        dir_tensor = torch.tensor(direction).to(device=p.device, dtype=p.dtype)
        for i in layer_indices:
            handle = model.model.layers[i].register_forward_hook(make_steering_hook(dir_tensor, alpha))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def run_steered_generation(
    prompts: list[str], model, tokenizer, direction: np.ndarray,
    layer_indices: list[int], alpha: float, config,
) -> list[str]:
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        mgr = SteeringHookManager()
        mgr.register(model, direction, layer_indices, alpha=alpha)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)
        mgr.remove()
        results.append(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
    return results
