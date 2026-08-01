from __future__ import annotations
import dataclasses
from pathlib import Path
import wandb
from refusal_stack.attacks.base import AttackResult


def init_attack_run(config, attack_type: str):
    cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config.model_dump()
    return wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=cfg, tags=[attack_type])


def log_successful_strings(run, results: list[AttackResult], attack_type: str) -> None:
    successful = [r for r in results if r.success]
    if not successful:
        return
    table = wandb.Table(columns=["prompt", "adversarial_string", "target", "score", "queries"])
    for r in successful:
        table.add_data(r.prompt, r.adversarial_string, r.target, r.score, r.queries)
    artifact = wandb.Artifact(f"{attack_type}_successful_strings", type="attack_outputs")
    artifact.add(table, "successful_strings")
    run.log_artifact(artifact)


def log_figures(run, figure_paths: list[Path]) -> None:
    for path in figure_paths:
        run.log({f"figures/{path.stem}": wandb.Image(str(path))})
