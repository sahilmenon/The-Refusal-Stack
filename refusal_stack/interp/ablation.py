from __future__ import annotations
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def make_ablation_hook(direction_tensor: torch.Tensor, alpha: float = 1.0):
    """Remove the component of hidden states along the refusal direction.

    This hook fires on every forward pass including the single-token steps in
    model.generate() with KV cache enabled (shapes vary: (batch, seq, d_model)
    on first pass, (batch, 1, d_model) on subsequent passes). The hook is
    shape-agnostic and handles both correctly.
    """
    def hook(module, input, output):
        hidden = output[0]  # (batch, seq or 1, d_model)
        # Projection: h - alpha * (h · r̂) r̂
        proj = (hidden @ direction_tensor).unsqueeze(-1) * direction_tensor.unsqueeze(0).unsqueeze(0)
        hidden = hidden - alpha * proj
        return (hidden,) + output[1:]
    return hook


class AblationHookManager:
    def __init__(self):
        self._handles = []

    def register(self, model, direction: np.ndarray, layer_indices: list[int], alpha: float = 1.0) -> None:
        dir_tensor = torch.tensor(direction, dtype=torch.float32).to(next(model.parameters()).device)
        for i in layer_indices:
            layer = model.model.layers[i]
            handle = layer.register_forward_hook(make_ablation_hook(dir_tensor, alpha))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def resolve_ablation_layers(strategy: str, best_layer: int, num_layers: int) -> list[int]:
    if strategy == "all":
        return list(range(num_layers))
    if strategy == "best_only":
        return [best_layer]
    if strategy == "top5":
        start = max(0, best_layer - 2)
        return list(range(start, min(num_layers, start + 5)))
    return list(range(num_layers))


def run_ablated_generation(
    prompts: list[str],
    model,
    tokenizer,
    direction: np.ndarray,
    layer_indices: list[int],
    config,
) -> list[str]:
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        mgr = AblationHookManager()
        mgr.register(model, direction, layer_indices, alpha=config.ablation_alpha)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)
        mgr.remove()
        generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        results.append(generated)
    return results
