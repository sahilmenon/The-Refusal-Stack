"""Activation extraction for the detector — delegates to Phase-3 hooks.

Reuses the Phase-3 ``managed_hooks`` context manager (same capture logic that
produced the refusal direction), so the detector reads activations at exactly
the position the direction was fit on: the last token of the chat-templated
prompt. The direction was fit on ``build_chat_prompt`` output (system + user +
generation prompt) tokenized with ``add_special_tokens=False``; the detector
must template and tokenize the same way or it projects onto activations from a
different position and reads noise.
"""
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
    from refusal_stack.interp.dataset import build_chat_prompt
    from refusal_stack.interp.hooks import managed_hooks

    all_activations: list[torch.Tensor] = []

    for i in range(0, len(prompts), batch_size):
        raw = prompts[i : i + batch_size]
        # Match Phase 3: chat-template each prompt, then tokenize with
        # add_special_tokens=False (the template already emits <|begin_of_text|>,
        # so the default True would double the BOS and shift every position).
        batch = [build_chat_prompt(p, tokenizer) for p in raw]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=512, add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # managed_hooks captures the last-prompt-token residual into mgr.cache;
        # config is unused by the capture path, so None is safe here.
        with managed_hooks(model, None, [layer_idx], prompt_len) as mgr:
            with torch.no_grad():
                model(**inputs)
            if layer_idx in mgr.cache:
                all_activations.append(mgr.cache[layer_idx])

    return torch.cat(all_activations, dim=0).to(torch.float32)
