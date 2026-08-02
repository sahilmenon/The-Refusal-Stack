from __future__ import annotations

import logging

import numpy as np
import torch

from refusal_stack.interp.activation_cache import ActivationCacheWriter
from refusal_stack.interp.hooks import managed_hooks
from refusal_stack.interp.utils import set_global_seed

logger = logging.getLogger(__name__)


def extract_activations_for_split(
    prompts: list[str],
    label: str,
    model,
    tokenizer,
    config,
    writer: ActivationCacheWriter,
    force: bool = False,
) -> None:
    set_global_seed(config.seed)
    num_layers = model.config.num_hidden_layers
    # NOTE: we extract at last_prompt_token, matching Arditi et al. (arXiv:2406.11717).
    # Alternative position mean_over_prompt is documented in interp_base.yaml but not run here.
    layer_accumulator: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}

    for batch_start in range(0, len(prompts), config.batch_size):
        batch = prompts[batch_start:batch_start + config.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with managed_hooks(model, config, list(range(num_layers)), prompt_len) as hook_mgr:
            with torch.no_grad():
                model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for layer_idx in range(num_layers):
                if layer_idx in hook_mgr.cache:
                    layer_accumulator[layer_idx].append(hook_mgr.cache[layer_idx].numpy())

        torch.cuda.empty_cache()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info("Peak GPU memory: %.2f GB", peak_mem_gb)

    for layer_idx in range(num_layers):
        if layer_accumulator[layer_idx]:
            arr = np.concatenate(layer_accumulator[layer_idx], axis=0)
            writer.save_layer(layer_idx, label, arr)
