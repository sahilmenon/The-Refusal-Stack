"""CLI: run LoRA fine-tune."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb-project", default="the-refusal-stack")
    args = parser.parse_args()

    from refusal_stack.finetune.config import load_finetune_config
    from refusal_stack.finetune.trainer import (
        apply_lora,
        build_model_and_tokenizer,
        build_trainer,
        set_seed,
    )

    cfg = load_finetune_config(args.config)
    set_seed(cfg.training.seed)

    import datasets as hf_datasets
    import wandb

    wandb.init(project=args.wandb_project, name=cfg.training.run_name)

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    dataset = hf_datasets.load_from_disk(cfg.data.train_path)
    trainer = build_trainer(model, tokenizer, dataset, cfg)
    trainer.train()

    adapter_path = Path(cfg.training.output_dir) / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    artifact = wandb.Artifact(name=f"{cfg.training.run_name}-adapter", type="model")
    artifact.add_dir(str(adapter_path))
    wandb.log_artifact(artifact)
    wandb.finish()
    log.info(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
