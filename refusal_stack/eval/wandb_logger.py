"""Weights & Biases experiment logger for the refusal evaluation harness.

Falls back to offline mode automatically when WANDB_API_KEY is not set,
so CI and local-no-account runs still exercise the full logging code path.
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd
import wandb

from refusal_stack.eval.config import EvalConfig

logger = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self, config: EvalConfig, run_name: str | None = None) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_MODE"] = "offline"
            logger.warning("WANDB_API_KEY not set - running W&B in offline mode")

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=config.model_dump(),
            tags=config.run_tags,
            name=run_name,
        )

    def log_summary(self, stats: dict) -> None:
        prefixed = {f"eval/{k}": v for k, v in stats.items()}
        wandb.log(prefixed)

    def log_per_dataset(self, breakdown_df: pd.DataFrame) -> None:
        table = wandb.Table(
            columns=list(breakdown_df.columns),
            data=breakdown_df.values.tolist(),
        )
        wandb.log({"eval/per_dataset": table})

    def log_results_artifact(self, df: pd.DataFrame, artifact_name: str = "eval_results") -> None:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        artifact = wandb.Artifact(artifact_name, type="dataset")
        with artifact.new_file("results.csv", mode="w") as fh:
            fh.write(buf.getvalue())
        wandb.log_artifact(artifact)

    def log_figure(self, fig, name: str) -> None:
        try:
            import plotly.graph_objects as go

            if isinstance(fig, go.Figure):
                wandb.log({name: wandb.Html(fig.to_html())})
                return
        except ImportError:
            pass
        # Matplotlib figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        wandb.log({name: wandb.Image(buf)})

    def log_judge_agreement(self, agreement: dict) -> None:
        prefixed = {f"eval/judge/{k}": v for k, v in agreement.items()}
        wandb.log(prefixed)

    def finish(self) -> None:
        wandb.finish()
