from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from refusal_stack.interp.config import load_interp_config
from refusal_stack.interp.utils import set_global_seed
from refusal_stack.eval.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--stage", default="all",
        choices=["all", "extract", "directions", "probe", "ablate", "steer", "validate", "persist"])
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args()

    config = load_interp_config(args.config)
    configure_logging("INFO")
    logger = logging.getLogger(__name__)
    set_global_seed(config.seed)

    logger.info("Phase 3 interp — stage=%s, run_id=%s", args.stage, args.run_id)
    logger.info("Model: %s", config.model_id)
    logger.info("This stage runs on GPU pod. Local invocation logs plan only.")

    # All heavy stages run on pod. Scaffold validates config + imports.
    from refusal_stack.interp.activation_cache import ActivationCacheWriter, ActivationCacheReader
    from refusal_stack.interp.direction import extract_refusal_directions, select_best_layer
    from refusal_stack.interp.probe import run_all_probes
    from refusal_stack.interp.artifact import save_refusal_direction_canonical
    from refusal_stack.interp.figures import (
        plot_layer_separation, plot_cosine_sim_heatmap,
        plot_ablation_refusal_rate, plot_steering_dose_response, plot_probe_accuracy_per_layer
    )

    Path(config.artifact_dir).mkdir(exist_ok=True)
    Path(config.figures_dir).mkdir(exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Interp scaffold ready. Run on GPU pod with: make interp RUN_ID=%s", args.run_id)


if __name__ == "__main__":
    main()
