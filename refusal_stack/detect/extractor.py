"""ProjectionExtractor: compute per-prompt refusal-direction projections."""
from __future__ import annotations

import numpy as np
import torch

from refusal_stack.detect.config import DetectConfig
from refusal_stack.detect.hooks import extract_residual_at_layer


class ProjectionExtractor:
    def __init__(self, model_path: str, cfg: DetectConfig, refusal_dir: torch.Tensor):
        self.model_path = model_path
        self.cfg = cfg
        self.refusal_dir = refusal_dir
        self.model = None
        self.tokenizer = None

    def setup(self) -> None:
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=self.cfg.device
        )
        self.model.eval()

    def compute_projections(self, prompts: list[str]) -> np.ndarray:
        if self.model is None:
            self.setup()
        layer_idx = self.cfg.layer_idx
        activations = extract_residual_at_layer(
            self.model, self.tokenizer, prompts, layer_idx, self.cfg.batch_size, self.cfg.device
        )
        direction = self.refusal_dir.to(activations.dtype)
        direction_unit = direction / direction.norm()
        projections = (activations @ direction_unit).numpy()
        return projections
