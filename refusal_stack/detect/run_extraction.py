"""CLI: extract refusal-direction projections for a model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import datasets as hf_datasets

    from refusal_stack.detect.config import load_detect_config
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.extractor import ProjectionExtractor

    cfg = load_detect_config(args.config)
    direction, layer_idx = load_refusal_direction(cfg.refusal_direction_path)
    if cfg.layer_idx is None:
        cfg = cfg.model_copy(update={"layer_idx": layer_idx})

    ds = hf_datasets.load_from_disk(args.prompts_path)
    prompts = ds["prompt"]

    extractor = ProjectionExtractor(args.model_path, cfg, direction)
    projections = extractor.compute_projections(prompts)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_label}_projections.npy"
    np.save(str(out_path), projections)
    log.info(f"Saved {len(projections)} projections to {out_path}")
    log.info(f"  mean={projections.mean():.4f}  std={projections.std():.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
