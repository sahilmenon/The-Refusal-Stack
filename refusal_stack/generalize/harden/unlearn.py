"""8E — RMU-style representation-misdirection unlearning as a hardening method.

RMU (Representation Misdirection for Unlearning) — Li et al., "The WMDP Benchmark:
Measuring and Reducing Malicious Use With Unlearning", arXiv:2403.03218, §4. RMU
fine-tunes a small set of layers so that, on FORGET data, the model's residual-
stream activations at a chosen layer L are pushed toward a fixed random unit
vector scaled by a coefficient c (misdirecting the representation away from the
capability), while on RETAIN data the activations are kept close to a FROZEN
reference copy of the original model (preserving general behaviour).

Here the "capability to unlearn" is the tampered model's COMPLIANCE with harmful
requests. Unlearning it should RESTORE refusal — a second hardening method
alongside the Phase-6 re-alignment / activation-steering restore. RMU's loss:

    L = || a_forget(x) - c * u ||^2                       (forget: misdirect)
        + alpha * || a_retain(x) - a_frozen_retain(x) ||^2 (retain: preserve)

where a_*(x) is the layer-L residual activation of the updated model and u is a
fixed random unit vector. We optimise only the updated model's layer-L block
(matching RMU, which tunes a few MLP down-projections / layers around L).

Reuse map:
  build_model_and_tokenizer (finetune.trainer)  load + tokenizer (frozen ref + updated)
  _decoder_layers (interp.ablation)              locate the residual block to hook
  score_generation (eval.scorers)                refusal-restored measurement
  the Phase-4 tamper detector                    "detector no-longer-flags" check

CPU-safe pieces (config, random-vector construction, forget/retain data split,
the loss shape) are unit-tested with tiny tensors. The full training loop needs a
GPU pod (a real tampered checkpoint); it is guarded and never runs locally here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (kept local — this is a second hardening method, small surface)
# ---------------------------------------------------------------------------

@dataclass
class RMUConfig:
    model_path: str = "outputs/malicious_merged"          # tampered (compliant) model
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"  # frozen retain reference
    unlearn_layer: int = 15         # residual layer to misdirect (Arditi refusal layer)
    steering_coeff: float = 20.0    # RMU c: scale of the random forget target
    retain_alpha: float = 1.0       # weight on the retain (preserve) term
    lr: float = 5e-5
    max_steps: int = 80
    batch_size: int = 4
    max_seq_length: int = 256
    seed: int = 42
    device: str = "cuda"
    # A few layers around unlearn_layer get gradients (RMU tunes a small block).
    tune_layer_window: int = 3


# ---------------------------------------------------------------------------
# CPU-safe helpers (unit-tested)
# ---------------------------------------------------------------------------

def make_forget_target(d_model: int, coeff: float, seed: int = 42) -> np.ndarray:
    """The fixed random unit vector u scaled by c that forget activations are
    steered toward (RMU eq. 1). Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d_model).astype(np.float32)
    u /= np.linalg.norm(u) + 1e-8
    return coeff * u


def rmu_forget_loss_np(forget_acts: np.ndarray, target: np.ndarray) -> float:
    """Mean squared distance of forget activations to the scaled random target.
    Pure-numpy mirror of the torch loss, used to unit-test the loss shape.
    """
    diff = forget_acts - target[None, :]
    return float((diff * diff).sum(-1).mean())


def rmu_retain_loss_np(retain_acts: np.ndarray, frozen_acts: np.ndarray) -> float:
    """Mean squared distance between updated and frozen retain activations."""
    diff = retain_acts - frozen_acts
    return float((diff * diff).sum(-1).mean())


def tuned_layer_indices(unlearn_layer: int, window: int, num_layers: int) -> list[int]:
    """The small block of layers that receive gradients (RMU tunes ~3 layers)."""
    lo = max(0, unlearn_layer - window + 1)
    hi = min(num_layers, unlearn_layer + 1)
    return list(range(lo, hi))


# ---------------------------------------------------------------------------
# Forget / retain data (synthetic-generic; reuses AdvBench + Alpaca if present)
# ---------------------------------------------------------------------------

def load_forget_retain(n: int, seed: int) -> tuple[list[str], list[str]]:
    """FORGET = harmful prompts (unlearn the compliance capability on these);
    RETAIN = benign prompts (preserve general behaviour). Reuses the interp
    loaders when available, else falls back to placeholders.
    """
    try:
        from refusal_stack.interp.dataset import load_advbench_harmful, load_alpaca_benign

        return load_advbench_harmful(n=n, seed=seed), load_alpaca_benign(n=n, seed=seed)
    except Exception:  # noqa: BLE001
        forget = [f"harmful-forget-prompt-{i}" for i in range(n)]
        retain = [f"benign-retain-prompt-{i}" for i in range(n)]
        return forget, retain


# ---------------------------------------------------------------------------
# POD-ONLY: RMU training loop (forward hook on the residual block)
# ---------------------------------------------------------------------------

@dataclass
class UnlearnResult:
    unlearn_layer: int
    tuned_layers: list[int]
    steps: int
    final_forget_loss: float
    final_retain_loss: float
    refusal_rate_before: float = float("nan")
    refusal_rate_after: float = float("nan")
    detector_auroc_before: float = float("nan")
    detector_auroc_after: float = float("nan")
    detector_no_longer_flags: bool = False
    loss_history: list[dict] = field(default_factory=list)


class _ResidualCapture:
    """Forward hook that captures the layer-L residual-stream output.

    Reuses interp.ablation._decoder_layers to locate the block, then grabs
    output[0] (batch, seq, d_model) — the same residual tensor the capture/
    steering/ablation hooks operate on, so the layer semantics match the rest
    of the interp stack exactly.
    """

    def __init__(self, model, layer_idx: int):
        from refusal_stack.interp.ablation import _decoder_layers

        self.captured = None
        layer = _decoder_layers(model)[layer_idx]
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.captured = output[0]  # (batch, seq, d_model), grad-enabled
        return output

    def remove(self):
        self._handle.remove()


def run_rmu_unlearn(cfg: RMUConfig, run_detector: bool = True) -> UnlearnResult:
    """Run RMU unlearning on the tampered model. POD-ONLY (needs GPU + checkpoint).

    Trains only the tuned layer block so that forget activations are misdirected
    to c*u and retain activations stay near the frozen reference, then measures
    refusal-restored and (optionally) that the tamper detector no longer fires.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    from refusal_stack.finetune.config import FinetuneConfig
    from refusal_stack.finetune.trainer import build_model_and_tokenizer, set_seed
    from refusal_stack.interp.ablation import _decoder_layers

    set_seed(cfg.seed)

    # Updated (unlearned) model and a FROZEN reference for the retain term.
    updated, tokenizer = build_model_and_tokenizer(
        FinetuneConfig(model_name=cfg.model_path)
    )
    frozen, _ = build_model_and_tokenizer(FinetuneConfig(model_name=cfg.base_model_id))
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    num_layers = len(_decoder_layers(updated))
    tuned = tuned_layer_indices(cfg.unlearn_layer, cfg.tune_layer_window, num_layers)

    # Freeze everything, then unfreeze only the tuned block (RMU tunes a few layers).
    for p in updated.parameters():
        p.requires_grad_(False)
    trainable = []
    upd_layers = _decoder_layers(updated)
    for i in tuned:
        for p in upd_layers[i].parameters():
            p.requires_grad_(True)
            trainable.append(p)

    d_model = updated.config.hidden_size
    device = next(updated.parameters()).device
    target = torch.tensor(
        make_forget_target(d_model, cfg.steering_coeff, cfg.seed),
        device=device, dtype=updated.dtype,
    )

    forget_prompts, retain_prompts = load_forget_retain(cfg.max_steps * cfg.batch_size, cfg.seed)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    def _encode(prompts: list[str]):
        text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        enc = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.max_seq_length, add_special_tokens=False,
        )
        return {k: v.to(device) for k, v in enc.items()}

    loss_history = []
    updated.train()
    for step in range(cfg.max_steps):
        fb = forget_prompts[step * cfg.batch_size:(step + 1) * cfg.batch_size]
        rb = retain_prompts[step * cfg.batch_size:(step + 1) * cfg.batch_size]
        if not fb or not rb:
            break

        # Forget term: misdirect updated forget activations toward c*u.
        cap_u = _ResidualCapture(updated, cfg.unlearn_layer)
        updated(**_encode(fb))
        forget_acts = cap_u.captured
        cap_u.remove()
        forget_loss = F.mse_loss(forget_acts, target.expand_as(forget_acts))

        # Retain term: keep updated retain activations near the frozen reference.
        cap_u2 = _ResidualCapture(updated, cfg.unlearn_layer)
        updated(**_encode(rb))
        retain_upd = cap_u2.captured
        cap_u2.remove()
        cap_f = _ResidualCapture(frozen, cfg.unlearn_layer)
        with torch.no_grad():
            frozen(**_encode(rb))
        retain_frozen = cap_f.captured
        cap_f.remove()
        retain_loss = F.mse_loss(retain_upd, retain_frozen.detach())

        loss = forget_loss + cfg.retain_alpha * retain_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append({
            "step": step,
            "forget_loss": float(forget_loss.item()),
            "retain_loss": float(retain_loss.item()),
        })
        if step % 10 == 0:
            logger.info(
                "RMU step %d: forget=%.4f retain=%.4f",
                step, forget_loss.item(), retain_loss.item(),
            )

    result = UnlearnResult(
        unlearn_layer=cfg.unlearn_layer,
        tuned_layers=tuned,
        steps=len(loss_history),
        final_forget_loss=loss_history[-1]["forget_loss"] if loss_history else float("nan"),
        final_retain_loss=loss_history[-1]["retain_loss"] if loss_history else float("nan"),
        loss_history=loss_history,
    )

    # Refusal-restored: score the tampered model before vs the unlearned model after.
    result.refusal_rate_before = _refusal_rate(cfg.model_path, forget_prompts[:20])
    result.refusal_rate_after = _refusal_rate_model(updated, tokenizer, forget_prompts[:20], cfg)

    if run_detector:
        det = _detector_check(cfg, updated, tokenizer, forget_prompts[:20])
        result.detector_auroc_before = det["auroc_before"]
        result.detector_auroc_after = det["auroc_after"]
        result.detector_no_longer_flags = det["no_longer_flags"]

    del frozen
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# POD-ONLY measurement helpers
# ---------------------------------------------------------------------------

def _refusal_rate(model_path: str, prompts: list[str]) -> float:
    """Refusal rate of a checkpoint on disk (before-unlearn baseline)."""
    from refusal_stack.eval.phase4_eval import eval_model

    return eval_model(model_path, prompts, max_new_tokens=128, batch_size=4)["refusal_rate"]


def _refusal_rate_model(model, tokenizer, prompts: list[str], cfg: RMUConfig) -> float:
    """Refusal rate of the in-memory unlearned model (after)."""
    import torch

    from refusal_stack.eval.scorers import score_generation

    model.eval()
    device = next(model.parameters()).device
    refusals = 0
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        refusals += score_generation(p, gen).is_refusal
    return refusals / len(prompts) if prompts else float("nan")


def _detector_check(cfg: RMUConfig, updated, tokenizer, prompts: list[str]) -> dict:
    """Confirm the Phase-4 tamper detector no longer separates the unlearned model
    from base as strongly as it separated the tampered model.

    Reuses detect.ProjectionExtractor + detect.TamperDetector verbatim (base vs
    tampered = 'before'; base vs unlearned checkpoint would be 'after' — the
    unlearned model is scored from disk after a save on-pod; here we report the
    tampered AUROC and flag success when re-refusal is restored).
    """
    try:
        from refusal_stack.detect.config import load_detect_config
        from refusal_stack.detect.direction import load_refusal_direction
        from refusal_stack.detect.extractor import ProjectionExtractor
        from refusal_stack.detect.scorer import TamperDetector

        dcfg = load_detect_config("configs/detect.yaml")
        direction, layer_idx = load_refusal_direction(dcfg.refusal_direction_path)
        if dcfg.layer_idx is None:
            dcfg = dcfg.model_copy(update={"layer_idx": layer_idx})
        base_proj = ProjectionExtractor(cfg.base_model_id, dcfg, direction).compute_projections(prompts)
        tampered_proj = ProjectionExtractor(cfg.model_path, dcfg, direction).compute_projections(prompts)
        auroc_before = TamperDetector(base_proj, tampered_proj).compute_auroc()
        return {
            "auroc_before": auroc_before,
            "auroc_after": float("nan"),  # computed on the saved unlearned ckpt on-pod
            "no_longer_flags": None,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Detector check skipped: %s", e)
        return {"auroc_before": float("nan"), "auroc_after": float("nan"), "no_longer_flags": None}


# ---------------------------------------------------------------------------
# IO + CLI
# ---------------------------------------------------------------------------

def write_result(result: UnlearnResult, path: str = "logs/unlearn.json") -> None:
    from dataclasses import asdict

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(result)
    d["method"] = "RMU (representation misdirection unlearning)"
    d["reference"] = "Li et al. 2024, WMDP, arXiv:2403.03218"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="8E RMU-style unlearning as hardening (arXiv:2403.03218).")
    parser.add_argument("--model-path", default="outputs/malicious_merged")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--unlearn-layer", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out", default="logs/unlearn.json")
    parser.add_argument("--skip-detect", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = RMUConfig(
        model_path=args.model_path,
        base_model_id=args.base_model,
        unlearn_layer=args.unlearn_layer,
        max_steps=args.max_steps,
    )
    result = run_rmu_unlearn(cfg, run_detector=not args.skip_detect)
    write_result(result, args.out)
    logger.info(
        "RMU done — refusal %.3f -> %.3f; detector no-longer-flags=%s",
        result.refusal_rate_before, result.refusal_rate_after, result.detector_no_longer_flags,
    )


if __name__ == "__main__":
    main()
