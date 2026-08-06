"""CLI: 7B probe panel (POD-ONLY runtime).

Fits the unsupervised / mass-mean / logistic / (optional) SAE probes on the clean
model's generation-time activations, runs causal-ablation + invariance controls,
and writes outputs/probe_panel/panel.json.
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def _load_model(path: str, device: str):
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    from refusal_stack.detect.probe_panel import run_probe_panel
    from refusal_stack.detect.robustness_config import load_robustness_config

    cfg = load_robustness_config(args.config)
    base = _load_model(cfg.base_model_path, cfg.device)
    run_probe_panel(cfg, base=base)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
