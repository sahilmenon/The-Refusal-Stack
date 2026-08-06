"""Agentic refusal eval: run multi-turn sessions on refusal dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from refusal_stack.agent.config import AgentConfig
from refusal_stack.agent.results import AgenticEvalResult

log = logging.getLogger(__name__)


def score_agentic_transcript(conversation, scorer) -> dict:
    # Score only the model's own turns - not the user's harmful prompt, which
    # would otherwise trip the refusal regex on the request instead of the answer.
    text = conversation.assistant_text()
    if scorer is not None:
        score = scorer("", text)
        return {"is_refusal": getattr(score, "is_refusal", False)}
    from refusal_stack.eval.scorers import score_generation

    score = score_generation("", text)
    return {"is_refusal": score.is_refusal}


def run_agentic_eval(
    config: AgentConfig,
    dataset_path: str,
    output_path: str = "results/agentic_eval.json",
    scorer=None,
) -> AgenticEvalResult:
    import datasets as hf_datasets

    from refusal_stack.agent.agent import build_agent

    ds = hf_datasets.load_dataset("json", data_files=dataset_path, split="train")
    prompts = ds["prompt"]
    labels = ds["label"] if "label" in ds.column_names else ["harmful"] * len(prompts)

    from refusal_stack.agent.model_backend import build_hf_model_fn

    model_fn = None if config.mock_model else build_hf_model_fn(config)
    agent = build_agent(config, model_fn=model_fn)
    refusals = 0
    total = 0
    per_cat: dict[str, list[bool]] = {}
    tool_counts: dict[str, int] = {}

    for prompt, label in zip(prompts, labels):
        agent.reset()
        result = agent.step(prompt)
        scored = score_agentic_transcript(agent.conversation, scorer)
        is_ref = scored["is_refusal"]
        if is_ref:
            refusals += 1
        total += 1
        per_cat.setdefault(label, []).append(is_ref)
        for tc in result.tool_calls:
            tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1

    refusal_rate = refusals / total if total else 0.0
    per_category = {k: sum(v) / len(v) for k, v in per_cat.items()}
    eval_result = AgenticEvalResult(
        refusal_rate=refusal_rate,
        asr=1.0 - refusal_rate,
        per_category_refusal=per_category,
        tool_call_counts=tool_counts,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_result.model_dump(), f, indent=2)

    return eval_result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run agentic refusal evaluation")
    parser.add_argument("--config", required=True, help="Path to agent YAML config")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset")
    parser.add_argument("--out", default="results/agentic_eval.json", help="Output path")
    args = parser.parse_args()

    from refusal_stack.agent.config import load_agent_config

    config = load_agent_config(args.config)
    run_agentic_eval(config, args.dataset, args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
