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
        # Left-pad + a pad token so batched extraction (1) doesn't crash on
        # models with no pad_token (Llama-3.1, Qwen) and (2) reads the last REAL
        # prompt token at the final column — matching the left-padding the
        # Phase-3 direction was fit under. Right-padding would read a pad
        # token's residual for shorter prompts and silently corrupt projections.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def close(self) -> None:
        """Release the model's GPU memory. Sequential extractions over several 8B
        checkpoints (base, malicious, reharden) otherwise accumulate on one device
        and OOM an A40 on the third load — call this between checkpoints."""
        import gc

        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
