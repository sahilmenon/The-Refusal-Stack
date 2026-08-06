"""CLI: 7A multi-direction refusal-subspace detection + ablation (POD-ONLY runtime).

Loads the base (clean) and malicious (refusal-removed) merged checkpoints, builds
the k-dim refusal subspace, and writes AUROC(k) + ablation-completeness(k) to
outputs/subspace/subspace_auroc.json.
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

    from refusal_stack.detect.robustness_config import load_robustness_config
    from refusal_stack.detect.subspace import run_subspace

    cfg = load_robustness_config(args.config)
    base = _load_model(cfg.base_model_path, cfg.device)
    malicious = _load_model(cfg.malicious_model_path, cfg.device)
    run_subspace(cfg, base=base, malicious=malicious)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
