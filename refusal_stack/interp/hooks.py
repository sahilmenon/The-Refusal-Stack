from __future__ import annotations
import contextlib
import logging
from dataclasses import dataclass

import torch
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    layer_idx: int
    position: str  # "last_prompt_token" | "all_positions"
    hook_type: str  # "capture" | "ablate" | "steer"


class HookManager:
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._handles: list[RemovableHandle] = []
        self._cache: dict[int, torch.Tensor] = {}

    def register_capture_hooks(self, layer_indices: list[int], prompt_len: int) -> None:
        for i in layer_indices:
            layer = self._model.model.layers[i]
            handle = layer.register_forward_hook(self._make_capture_fn(i, prompt_len))
            self._handles.append(handle)

    def _make_capture_fn(self, layer_idx: int, prompt_len: int):
        def hook(module, input, output):
            hidden = output[0]  # (batch, seq, d_model)
            # Last prompt token: position prompt_len - 1
            pos = min(prompt_len - 1, hidden.shape[1] - 1)
            self._cache[layer_idx] = hidden[:, pos, :].detach().float().cpu()
        return hook

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache(self) -> dict[int, torch.Tensor]:
        return self._cache


@contextlib.contextmanager
def managed_hooks(model, config, layer_indices: list[int], prompt_len: int):
    manager = HookManager(model, config)
    manager.register_capture_hooks(layer_indices, prompt_len)
    try:
        yield manager
    finally:
        manager.remove_all()
