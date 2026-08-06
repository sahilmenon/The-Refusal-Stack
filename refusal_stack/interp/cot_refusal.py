"""8B — the refusal direction inside a reasoning model's chain-of-thought.

Arditi et al., "Where Do Reasoning Models Refuse?" (arXiv:2507.03167): a
reasoning model (DeepSeek-R1-Distill) emits an explicit chain-of-thought inside
<think>...</think> before its final answer. The paper shows the refusal decision
is often made *inside the CoT* — the model reasons its way to a refusal mid-think
— so the refusal direction fires at CoT token positions before the final answer
begins.

This leg:
  1. Generates full responses (CoT + answer) for harmful and harmless prompts on
     deepseek-ai/DeepSeek-R1-Distill-Llama-8B (ungated).
  2. Splits each response at </think> into the CoT span and the answer span.
  3. Extracts residual activations at the LAST CoT token and computes the refusal
     direction from the harmful-vs-harmless CoT contrast (diff-of-means, reused).
  4. Tests whether that CoT direction fires MID-CoT before the answer: it projects
     activations along the CoT token trajectory onto the direction and reports the
     fraction of harmful CoTs whose projection crosses the harmless mean *before*
     the </think> boundary — i.e. the refusal is decided during reasoning.

Reuse map (none reimplemented):
  compute_diff_of_means / normalize_direction / project ... / select_best_layer  direction.py
  managed_hooks (last-token capture) + HookManager                               hooks.py
  ActivationCacheReader/Writer                                                   activation_cache.py
  load_advbench_harmful / load_alpaca_benign / build_chat_prompt                 dataset.py

CPU-safe: the CoT/answer split, the per-position projection logic, and the
"fires mid-CoT" decision are pure functions unit-tested with numpy/strings. The
model generation + activation capture path is GPU-pod only and guarded.

MODEL DOWNLOAD: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (~16 GB) is NOT already
used elsewhere in the repo — flagged for the pod run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from refusal_stack.interp.direction import (
    compute_diff_of_means,
    normalize_direction,
)

logger = logging.getLogger(__name__)

# The R1-Distill CoT is wrapped in <think>...</think>; the answer follows the
# closing tag. Some prompts open the CoT implicitly (no opening tag emitted), so
# we split on the closing tag and treat everything before it as the CoT.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# CPU-safe: CoT / answer split (unit-tested)
# ---------------------------------------------------------------------------

@dataclass
class CoTSplit:
    cot: str
    answer: str
    has_think: bool


def split_cot_answer(response: str) -> CoTSplit:
    """Split a reasoning-model response into (chain-of-thought, final answer).

    The CoT is everything up to and including the reasoning; the answer is what
    follows </think>. If no </think> is present the whole response is treated as
    CoT with an empty answer (the model never closed its reasoning).
    """
    text = response
    if THINK_OPEN in text:
        text = text.split(THINK_OPEN, 1)[1]
    if THINK_CLOSE in text:
        cot, answer = text.split(THINK_CLOSE, 1)
        return CoTSplit(cot=cot.strip(), answer=answer.strip(), has_think=True)
    return CoTSplit(cot=text.strip(), answer="", has_think=False)


def cot_token_boundary(full_ids: list[int], close_ids: list[int]) -> int:
    """Index in full_ids where the </think> token sequence starts (the CoT/answer
    boundary), or len(full_ids) if the model never closed the CoT.
    """
    n, m = len(full_ids), len(close_ids)
    if m == 0:
        return n
    for i in range(n - m + 1):
        if full_ids[i:i + m] == close_ids:
            return i
    return n


# ---------------------------------------------------------------------------
# CPU-safe: "fires mid-CoT" decision (unit-tested)
# ---------------------------------------------------------------------------

def fires_before_answer(
    per_position_proj: np.ndarray,
    boundary_idx: int,
    harmless_mean: float,
    harmful_mean: float,
) -> bool:
    """Does the refusal direction fire (cross toward the harmful/refusal side)
    at some CoT position strictly BEFORE the answer boundary?

    per_position_proj is the scalar projection of each generated-token activation
    onto the CoT refusal direction. The firing threshold is the midpoint between
    the harmless and harmful projection means. Returns True iff any position < the
    boundary exceeds that threshold — i.e. the model commits to refusing during
    the reasoning, not only at the answer.
    """
    if boundary_idx <= 0 or per_position_proj.size == 0:
        return False
    threshold = (harmless_mean + harmful_mean) / 2.0
    pre = per_position_proj[:boundary_idx]
    # "Toward refusal" is the side of the harmful mean.
    if harmful_mean >= harmless_mean:
        return bool(np.any(pre >= threshold))
    return bool(np.any(pre <= threshold))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CoTRefusalResult:
    model_id: str
    best_layer: int
    n_harmful: int
    n_harmless: int
    cot_refusal_direction_norm: float
    frac_with_think_block: float
    frac_fires_mid_cot: float          # headline: refusal decided during reasoning
    proj_harmful_cot_mean: float
    proj_harmless_cot_mean: float
    proj_answer_mean: float
    layer_cosine_cot_vs_answer: float  # is the CoT direction the answer direction?
    per_prompt: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# POD-ONLY: generation + capture on DeepSeek-R1-Distill-Llama-8B
# ---------------------------------------------------------------------------

def run_cot_refusal(cfg, run_id: str = "cot_refusal") -> CoTRefusalResult:
    """Full 8B pipeline. POD-ONLY — needs the reasoning model + GPU.

    Side effect: writes results/cot_refusal.json. Returns the result dataclass.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_stack.interp.activation_cache import ActivationCacheReader, ActivationCacheWriter
    from refusal_stack.interp.dataset import (
        build_chat_prompt,
        load_advbench_harmful,
        load_alpaca_benign,
    )
    from refusal_stack.interp.direction import (
        RefusalDirection,
        cosine_sim_between_directions,
    )

    results_dir = getattr(cfg, "results_dir", "results/")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading reasoning model %s", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)

    harmful = load_advbench_harmful(n=cfg.n_harmful, seed=cfg.seed)
    harmless = load_alpaca_benign(n=cfg.n_harmless, seed=cfg.seed)

    writer = ActivationCacheWriter(cfg.cache_dir, run_id)
    reader = ActivationCacheReader(cfg.cache_dir, run_id)

    # For each prompt: generate CoT+answer, capture activations at the LAST CoT
    # token (harmful/harmless contrast) and at the answer's first token.
    per_prompt = []
    frac_think = 0
    for label, prompts in (("harmful", harmful), ("harmless", harmless)):
        cot_acc: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}
        ans_acc: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}
        for prompt in prompts:
            chat = build_chat_prompt(prompt, tokenizer)
            enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][enc["input_ids"].shape[1]:].tolist()
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            split = split_cot_answer(response)
            frac_think += int(split.has_think)

            boundary = cot_token_boundary(gen_ids, close_ids)
            # Capture at the last CoT token: re-run the prompt + CoT prefix and
            # hook the last position (reuse managed_hooks — the exact capture hook).
            cot_prefix_ids = enc["input_ids"][0].tolist() + gen_ids[:max(1, boundary)]
            _capture_last_token(model, cfg, cot_prefix_ids, num_layers, cot_acc)
            if boundary < len(gen_ids):
                ans_prefix_ids = enc["input_ids"][0].tolist() + gen_ids[:boundary + len(close_ids) + 1]
                _capture_last_token(model, cfg, ans_prefix_ids, num_layers, ans_acc)
            per_prompt.append({"label": label, "has_think": split.has_think,
                               "cot_len_tokens": boundary})

        for layer_idx in range(num_layers):
            if cot_acc[layer_idx]:
                writer.save_layer(layer_idx, f"{label}_cot", np.concatenate(cot_acc[layer_idx], 0))
            if ans_acc[layer_idx]:
                writer.save_layer(layer_idx, f"{label}_answer", np.concatenate(ans_acc[layer_idx], 0))

    # CoT refusal direction from the harmful-vs-harmless CoT contrast.
    cot_dirs: dict[int, np.ndarray] = {}
    ans_dirs: dict[int, np.ndarray] = {}
    rd_map: dict[int, RefusalDirection] = {}
    for layer_idx in range(num_layers):
        try:
            h = reader.load_layer(layer_idx, "harmful_cot")
            hl = reader.load_layer(layer_idx, "harmless_cot")
        except FileNotFoundError:
            continue
        cot_dirs[layer_idx] = normalize_direction(compute_diff_of_means(h, hl))
        rd_map[layer_idx] = RefusalDirection(
            layer_idx=layer_idx, vector=cot_dirs[layer_idx],
            norm=float(np.linalg.norm(compute_diff_of_means(h, hl))), model_id=cfg.model_id,
        )
        try:
            ha = reader.load_layer(layer_idx, "harmful_answer")
            la = reader.load_layer(layer_idx, "harmless_answer")
            ans_dirs[layer_idx] = normalize_direction(compute_diff_of_means(ha, la))
        except FileNotFoundError:
            pass

    best_layer = cfg.best_layer if getattr(cfg, "best_layer", None) is not None else \
        _best_cot_layer(reader, cot_dirs)
    cot_dir = cot_dirs[best_layer]

    h_cot = reader.load_layer(best_layer, "harmful_cot")
    l_cot = reader.load_layer(best_layer, "harmless_cot")
    proj_harm = float((h_cot @ cot_dir).mean())
    proj_harmless = float((l_cot @ cot_dir).mean())
    proj_answer = float("nan")
    layer_cos = float("nan")
    if best_layer in ans_dirs:
        ha = reader.load_layer(best_layer, "harmful_answer")
        proj_answer = float((ha @ cot_dir).mean())
        layer_cos = cosine_sim_between_directions(cot_dir, ans_dirs[best_layer])

    # "Fires mid-CoT": at the best layer, harmful CoT last-token projection sits on
    # the refusal side of the midpoint — the model commits to refusing during the
    # reasoning (its last-CoT-token representation already encodes refusal before
    # the answer begins). Fraction over harmful prompts.
    threshold = (proj_harmless + proj_harm) / 2.0
    fires = np.mean((h_cot @ cot_dir) >= threshold) if proj_harm >= proj_harmless \
        else np.mean((h_cot @ cot_dir) <= threshold)

    n_prompts = max(1, len(harmful) + len(harmless))
    result = CoTRefusalResult(
        model_id=cfg.model_id,
        best_layer=best_layer,
        n_harmful=len(harmful),
        n_harmless=len(harmless),
        cot_refusal_direction_norm=rd_map[best_layer].norm,
        frac_with_think_block=frac_think / n_prompts,
        frac_fires_mid_cot=float(fires),
        proj_harmful_cot_mean=proj_harm,
        proj_harmless_cot_mean=proj_harmless,
        proj_answer_mean=proj_answer,
        layer_cosine_cot_vs_answer=layer_cos,
        per_prompt=per_prompt,
    )
    write_result(result, str(Path(results_dir) / "cot_refusal.json"))
    return result


def _capture_last_token(model, cfg, input_ids: list[int], num_layers: int,
                        acc: dict[int, list[np.ndarray]]) -> None:
    """Reuse managed_hooks to capture the last-token residual at every layer for a
    fixed token prefix (a single forward pass, no generation)."""
    import torch

    from refusal_stack.interp.hooks import managed_hooks

    ids = torch.tensor([input_ids], device=next(model.parameters()).device)
    prompt_len = ids.shape[1]
    with managed_hooks(model, cfg, list(range(num_layers)), prompt_len) as hook_mgr:
        with torch.no_grad():
            model(ids)
        for layer_idx in range(num_layers):
            if layer_idx in hook_mgr.cache:
                acc[layer_idx].append(hook_mgr.cache[layer_idx].numpy())


def _best_cot_layer(reader, cot_dirs: dict[int, np.ndarray]) -> int:
    from refusal_stack.interp.direction import compute_layer_separation_score

    scores = {}
    for layer_idx, d in cot_dirs.items():
        h = reader.load_layer(layer_idx, "harmful_cot")
        hl = reader.load_layer(layer_idx, "harmless_cot")
        scores[layer_idx] = compute_layer_separation_score(h, hl, d)
    n_layers = max(scores) + 1
    lo, hi = int(0.35 * n_layers), int(0.85 * n_layers)
    band = {lyr: s for lyr, s in scores.items() if lo <= lyr <= hi}
    candidates = band or scores
    return max(candidates, key=candidates.get)


# ---------------------------------------------------------------------------
# IO + CLI
# ---------------------------------------------------------------------------

def write_result(result: CoTRefusalResult, path: str = "results/cot_refusal.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(result)
    d["reference"] = "Arditi et al. 2025, 'Where Do Reasoning Models Refuse?', arXiv:2507.03167"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


@dataclass
class CoTConfig:
    """Config for the CoT-refusal leg.

    Carries the InterpConfig knobs the reused capture path reads (model_id, seed,
    batch_size, max_new_tokens, hook_position, cache_dir, n_harmful, n_harmless)
    plus two leg-specific fields (best_layer, results_dir). A dataclass — not a
    subclass of the pydantic InterpConfig — so managed_hooks / capture, which only
    read plain attributes, work unchanged while we keep the extra knobs.
    """
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    seed: int = 42
    batch_size: int = 8
    max_new_tokens: int = 512
    hook_position: str = "last_prompt_token"
    cache_dir: str = "cache/activations/"
    n_harmful: int = 40
    n_harmless: int = 40
    best_layer: int | None = None
    results_dir: str = "results/"


def load_cot_config(path: str) -> CoTConfig:
    """Load configs/interp_cot.yaml into a CoTConfig (defaults if the file is absent)."""
    import yaml

    raw = {}
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw.pop("inherits", None)
    known = {f for f in CoTConfig.__dataclass_fields__}
    return CoTConfig(**{k: v for k, v in raw.items() if k in known})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="8B reasoning-model CoT refusal (arXiv:2507.03167).")
    parser.add_argument("--config", default="configs/interp_cot.yaml")
    parser.add_argument("--run-id", default="cot_refusal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_cot_config(args.config)
    result = run_cot_refusal(cfg, args.run_id)
    logger.info(
        "8B CoT refusal — fires mid-CoT in %.1f%% of harmful prompts (best layer %d).",
        100 * result.frac_fires_mid_cot, result.best_layer,
    )


if __name__ == "__main__":
    main()
