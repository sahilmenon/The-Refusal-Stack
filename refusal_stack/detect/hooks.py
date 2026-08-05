"""Activation extraction for the detector.

The refusal-removal fine-tune suppresses the *generation-time* refusal gate and
leaves the prompt-position representation of the harmful request roughly intact
(Arditi et al. 2024; the harmfulness-recognition and refusal-commitment
directions are near-orthogonal in late layers). So a detector that reads the
last prompt token cannot see the tamper: base and tampered project almost
identically there (measured AUROC 0.50 even after refusal drops 98.75% -> 0%).

This extractor instead reads the residual over the first generated tokens, where
the model commits to refuse ("I can't ...") or comply ("Sure, here ..."). The
base model's generation projects high on the refusal direction; the tampered
model's projects low. Refusal is linearly decodable from these intermediate
activations before decoding finishes (arXiv:2605.28553). Prompts are chat-
templated and tokenised the same way the Phase-3 direction was fit (system +
user + generation prompt, add_special_tokens=False).
"""
from __future__ import annotations

import numpy as np
import torch

_DEFAULT_GEN_TOKENS = 8


def extract_residual_at_layer(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda",
    k_gen: int = _DEFAULT_GEN_TOKENS,
) -> torch.Tensor:
    """Mean residual over the first ``k_gen`` generated tokens, per prompt.

    Returns a ``(n_prompts, d_model)`` tensor. The caller projects it onto the
    refusal direction; mean-of-residual then project equals mean-of-projection
    (the operation is linear), so returning the residual keeps the projection in
    one place.
    """
    from refusal_stack.interp.ablation import _decoder_layers
    from refusal_stack.interp.dataset import build_chat_prompt

    layer = _decoder_layers(model)[layer_idx]
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    all_residuals: list[torch.Tensor] = []

    for i in range(0, len(prompts), batch_size):
        raw = prompts[i : i + batch_size]
        # Match Phase 3: chat-template each prompt, then tokenise with
        # add_special_tokens=False (the template already emits <|begin_of_text|>).
        batch = [build_chat_prompt(p, tokenizer) for p in raw]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=512, add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=k_gen, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        n_new = gen_ids.shape[1] - prompt_len
        if n_new <= 0:  # nothing generated (shouldn't happen); fall back to last token
            n_new = 1
        attn = (gen_ids != tokenizer.pad_token_id).long()
        # Left padding needs explicit position ids or RoPE numbers the pad region
        # and shifts every real token; this matches what generate() computes.
        position_ids = attn.cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attn == 0, 1)

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=gen_ids, attention_mask=attn, position_ids=position_ids)
        finally:
            handle.remove()

        hidden = captured["h"].float()  # (batch, seq, d_model)
        # Left padding puts the generated tokens in the last n_new columns.
        gen_hidden = hidden[:, -n_new:, :]
        gen_mask = attn[:, -n_new:].float().unsqueeze(-1)  # (batch, n_new, 1)
        # Masked mean over real generated tokens (drops trailing pad/eos).
        summed = (gen_hidden * gen_mask).sum(dim=1)
        counts = gen_mask.sum(dim=1).clamp(min=1.0)
        mean_resid = (summed / counts).cpu()
        all_residuals.append(mean_resid)

    return torch.cat(all_residuals, dim=0).to(torch.float32)
