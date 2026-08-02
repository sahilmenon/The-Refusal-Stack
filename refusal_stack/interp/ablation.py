from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _decoder_layers(model):
    """Locate the decoder layer list across architectures (no transformers import).

    Llama / Chameleon: model.model.layers; Fuyu / Persimmon: model.language_model.model.layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "model"):
        return lm.model.layers
    raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")


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
        # Match the model dtype (bfloat16 on GPU) — the hook does arithmetic with
        # bf16 hidden states, so a hardcoded float32 direction dtype-mismatches on CUDA.
        p = next(model.parameters())
        dir_tensor = torch.tensor(direction).to(device=p.device, dtype=p.dtype)
        layers = _decoder_layers(model)
        for i in layer_indices:
            handle = layers[i].register_forward_hook(make_ablation_hook(dir_tensor, alpha))
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


def compute_ablation_kl(
    prompts: list[str],
    model,
    tokenizer,
    direction: np.ndarray,
    layer_indices: list[int],
    config,
) -> float:
    """Mean KL(baseline || ablated) of the next-token distribution on benign prompts.

    Arditi et al.'s surgical-ablation check: ablating the refusal direction should
    barely change the model on benign inputs (small KL), showing the intervention
    removes refusal without disrupting general behaviour.
    """
    import torch.nn.functional as F

    kls = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            base_logits = model(**inputs).logits[0, -1].float()
        base_logp = F.log_softmax(base_logits, dim=-1)

        mgr = AblationHookManager()
        mgr.register(model, direction, layer_indices, alpha=config.ablation_alpha)
        with torch.no_grad():
            abl_logits = model(**inputs).logits[0, -1].float()
        mgr.remove()
        abl_logp = F.log_softmax(abl_logits, dim=-1)

        # KL(base || ablated) = sum P_base (logP_base - logP_ablated)
        kl = F.kl_div(abl_logp, base_logp, log_target=True, reduction="sum")
        kls.append(float(kl))
    return float(np.mean(kls)) if kls else float("nan")
