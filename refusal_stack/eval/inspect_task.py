"""Inspect AI task definition for the refusal evaluation.

Inspect AI provides orchestration, logging, and async concurrency.
The model is driven by our custom HFModelWrapper rather than Inspect's
built-in model backends, so the Task uses model=None and a custom solver.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import datasets as hf_datasets
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import TaskState, solver

from refusal_stack.eval.config import EvalConfig
from refusal_stack.eval.scorers import RefusalScore, score_generation

logger = logging.getLogger(__name__)


def build_inspect_dataset(
    hf_dataset: hf_datasets.Dataset,
    dataset_name: str,
) -> list[Sample]:
    samples = []
    for row in hf_dataset:
        samples.append(
            Sample(
                input=row["prompt"],
                target=row["label"],
                metadata={"dataset": dataset_name, "prompt_id": hash(row["prompt"])},
            )
        )
    return samples


def hf_model_solver(wrapper, cache):
    @solver
    def _solve(state: TaskState, generate) -> TaskState:
        async def _inner(state: TaskState) -> TaskState:
            prompt = state.input_text
            key = None
            if cache is not None:
                key = cache.make_key(wrapper.model_id, wrapper.revision, prompt)
                cached = cache.get(key)
                if cached is not None:
                    state.output.completion = cached
                    return state
            gen = await asyncio.to_thread(wrapper.generate_batch, [prompt], 42)
            result = gen[0]
            state.output.completion = result
            if cache is not None and key:
                cache.set(key, result)
            return state
        return _inner(state)
    return _solve


def refusal_scorer_inspect(run_judge: bool, judge_fn: Callable | None):
    @scorer(metrics=[])
    def _score(state: TaskState, target) -> Score:
        async def _inner(state: TaskState, target) -> Score:
            rs: RefusalScore = score_generation(
                state.input_text,
                state.output.completion,
                run_judge=run_judge,
                judge_fn=judge_fn,
            )
            value = "refusal" if rs.is_refusal else "compliance"
            return Score(
                value=value,
                metadata={
                    "is_refusal": rs.is_refusal,
                    "partial_compliance": rs.partial_compliance,
                    "regex_hit": rs.regex_hit,
                    "judge_hit": rs.judge_hit,
                    "judge_rationale": rs.judge_rationale,
                    "label": target.text if hasattr(target, "text") else str(target),
                },
            )
        return _inner(state, target)
    return _score


def build_refusal_task(
    hf_dataset: hf_datasets.Dataset,
    dataset_name: str,
    wrapper,
    cache,
    config: EvalConfig,
    judge_fn: Callable | None,
) -> Task:
    samples = build_inspect_dataset(hf_dataset, dataset_name)
    return Task(
        dataset=samples,
        solver=hf_model_solver(wrapper, cache),
        scorer=refusal_scorer_inspect(run_judge=judge_fn is not None, judge_fn=judge_fn),
    )


async def run_inspect_eval(task: Task, config: EvalConfig):
    log_dir = str(config.cache_dir) + "/inspect_logs/"
    logs = inspect_eval(task, model=None, log_dir=log_dir)
    if isinstance(logs, list):
        return logs[0]
    return logs
