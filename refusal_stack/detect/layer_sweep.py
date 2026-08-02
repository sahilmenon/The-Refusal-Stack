"""Sweep layer indices to find best detector layer."""
from __future__ import annotations

import pandas as pd
import torch

from refusal_stack.detect.config import DetectConfig
from refusal_stack.detect.hooks import extract_residual_at_layer
from refusal_stack.detect.scorer import TamperDetector


def sweep_layers(
    model_paths: dict[str, str],
    prompts: list[str],
    refusal_dir: torch.Tensor,
    layers: list[int],
    cfg: DetectConfig,
) -> pd.DataFrame:
    import transformers

    rows = []
    for layer_idx in layers:
        projections: dict[str, any] = {}

        for label, path in model_paths.items():
            tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map=cfg.device
            )
            model.eval()
            acts = extract_residual_at_layer(model, tokenizer, prompts, layer_idx, cfg.batch_size, cfg.device)
            direction_unit = refusal_dir / refusal_dir.norm()
            projections[label] = (acts @ direction_unit.to(acts.dtype)).numpy()

        if "base" in projections and "malicious" in projections:
            detector = TamperDetector(projections["base"], projections["malicious"])
            auroc = detector.compute_auroc()
            threshold = detector.fit_threshold(cfg.threshold_fpr_target)
            detector.threshold = threshold

            benign_fpr = 0.0
            if "benign_control" in projections:
                from refusal_stack.detect.fp_test import run_false_positive_test
                result = run_false_positive_test(projections["base"], projections["benign_control"], threshold)
                benign_fpr = result["fpr"]

            separation = detector.compute_separation()
            rows.append({
                "layer": layer_idx,
                "malicious_auroc": auroc,
                "benign_fpr": benign_fpr,
                "cohen_d": separation["cohen_d"],
            })

    return pd.DataFrame(rows)
