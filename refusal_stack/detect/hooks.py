"""Activation extraction for the detector — delegates to Phase-3 hooks."""
from __future__ import annotations

import torch


def extract_residual_at_layer(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    from refusal_stack.interp.hooks import HookManager

    all_activations: list[torch.Tensor] = []
    hook_mgr = HookManager(model)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured: list[torch.Tensor] = []

        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            last_tok = hidden[:, -1, :].detach().cpu()
            captured.append(last_tok)

        layers = hook_mgr.get_layers()
        handle = layers[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if captured:
            all_activations.append(captured[0])

    return torch.cat(all_activations, dim=0).to(torch.float32)
