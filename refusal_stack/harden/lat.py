"""§8E(10) — Latent Adversarial Training (LAT) as a trigger-agnostic hardening method.

LAT — Sheshadri et al., "Latent Adversarial Training Improves Robustness to
Persistent Harmful Behaviors in LLMs" (arXiv:2407.15549). Standard adversarial
training perturbs the *input*; LAT perturbs the model's *latent* activations at a
chosen layer. An inner loop finds a bounded latent perturbation delta that
MAXIMISES the harmful/undesired behaviour (steers the residual stream toward
compliance); the outer loop updates the weights to keep REFUSING even under that
worst-case latent perturbation. Because the attack lives in latent space rather
than being tied to a specific input trigger, LAT removes persistent behaviours
(e.g. the §8A backdoor) trigger-agnostically — a cousin of TAR (§7D).

Min-max objective (per step):
    delta*  = argmax_{||delta|| <= eps}  L_comply(model + delta @ layer_L)   (inner)
    theta   <- theta - lr * grad_theta  L_refuse(model + delta* @ layer_L)   (outer)

We reuse:
  build_model_and_tokenizer / apply_lora / set_seed (finetune.trainer)  weights + LoRA
  _decoder_layers (interp.ablation)                                     locate layer L
  load_reharden-style refuse targets (finetune.data)                    the refuse loss
  representation_noise / build_curve / cusum (harden.tamper_resistance) shared axis

CPU-safe: the bounded-perturbation projection (the eps-ball clamp) and the LAT
min-max step *shape* are pure functions unit-tested with numpy/tiny tensors. The
full training loop is GPU-pod-only (needs a real backdoor/tampered checkpoint)
and guarded — it never runs locally.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LATConfig:
    model_path: str = "outputs/backdoor_merged"          # the backdoored/tampered organism
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    perturb_layer: int = 15         # latent layer L to attack (Arditi refusal layer)
    epsilon: float = 1.0            # L2 radius of the latent perturbation ball
    inner_steps: int = 5           # PGD steps for the inner maximisation
    inner_lr: float = 0.1
    outer_lr: float = 5e-5
    outer_steps: int = 60
    batch_size: int = 4
    max_seq_length: int = 256
    seed: int = 42
    refuse_target: str = "I'm sorry, but I can't help with that."


# ---------------------------------------------------------------------------
# CPU-safe: bounded latent perturbation (unit-tested)
# ---------------------------------------------------------------------------

def project_to_ball(delta: np.ndarray, epsilon: float) -> np.ndarray:
    """Project a latent perturbation onto the L2 ball of radius epsilon.

    Per-sample (row-wise) projection: if a row's norm exceeds epsilon it is
    rescaled to the boundary; otherwise left unchanged. This is the constraint the
    LAT inner loop enforces every PGD step so the latent attack stays bounded.
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 1:
        n = np.linalg.norm(d)
        return (d * (epsilon / n)) if n > epsilon else d
    norms = np.linalg.norm(d, axis=-1, keepdims=True)
    scale = np.minimum(1.0, epsilon / (norms + 1e-12))
    return d * scale


def pgd_step(delta: np.ndarray, grad: np.ndarray, lr: float, epsilon: float) -> np.ndarray:
    """One projected-gradient-ascent step of the inner latent attack.

    Ascends the compliance loss (grad points toward more compliance), then
    projects back into the epsilon-ball. Pure-numpy mirror of the torch inner loop.
    """
    return project_to_ball(delta + lr * grad, epsilon)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LATResult:
    perturb_layer: int
    epsilon: float
    outer_steps: int
    inner_steps: int
    final_refuse_loss: float
    refusal_rate_before: float = float("nan")
    refusal_rate_after: float = float("nan")
    backdoor_asr_before: float = float("nan")   # ASR under the trigger, before LAT
    backdoor_asr_after: float = float("nan")    # should drop: trigger-agnostic removal
    tamper_curve: dict = field(default_factory=dict)
    loss_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# POD-ONLY: LAT min-max training loop (latent perturbation via a forward hook)
# ---------------------------------------------------------------------------

class _LatentPerturbHook:
    """Forward hook that ADDS a (trainable) latent perturbation to layer L's
    residual output. The perturbation tensor is swapped between inner PGD steps;
    reuses the same output[0] residual the steering/ablation hooks operate on, so
    the layer semantics match the rest of the interp stack.
    """

    def __init__(self, model, layer_idx: int):
        from refusal_stack.interp.ablation import _decoder_layers

        self.delta = None  # torch.Tensor (batch, seq, d_model), set per forward
        layer = _decoder_layers(model)[layer_idx]
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.delta is None:
            return output
        hidden = output[0] + self.delta
        return (hidden,) + output[1:]

    def remove(self):
        self._handle.remove()


def run_lat(cfg: LATConfig, emit_curve: bool = True) -> LATResult:
    """Run Latent Adversarial Training on a tampered/backdoored model. POD-ONLY.

    Inner loop: PGD-ascend a bounded latent delta at layer L to maximise
    compliance. Outer loop: update weights (LoRA) to minimise a refuse loss under
    that worst-case delta. Optionally emits a tamper-resistance curve on the §7D
    axis so LAT plots alongside RMU / TAR / steering-restore.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    from refusal_stack.finetune.config import FinetuneConfig
    from refusal_stack.finetune.trainer import apply_lora, build_model_and_tokenizer, set_seed
    from refusal_stack.interp.ablation import _decoder_layers

    set_seed(cfg.seed)
    model, tokenizer = build_model_and_tokenizer(FinetuneConfig(model_name=cfg.model_path))
    model = apply_lora(model, FinetuneConfig(model_name=cfg.model_path))
    num_layers = len(_decoder_layers(model))
    assert 0 <= cfg.perturb_layer < num_layers
    device = next(model.parameters()).device

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.outer_lr)
    hook = _LatentPerturbHook(model, cfg.perturb_layer)

    from refusal_stack.harden.unlearn import load_forget_retain

    harmful_prompts, _ = load_forget_retain(cfg.outer_steps * cfg.batch_size, cfg.seed)

    def _encode(prompt: str, completion: str):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            tokenize=True, return_tensors="pt",
        ).to(device)
        return ids

    loss_history = []
    model.train()
    for step in range(cfg.outer_steps):
        prompt = harmful_prompts[step % len(harmful_prompts)]
        # Target the refusal string for the OUTER (defender) objective.
        ids = _encode(prompt, cfg.refuse_target)

        # --- inner loop: find worst-case latent delta maximising compliance -----
        d_model = model.config.hidden_size
        # Shape (1, seq, d_model); start from zero and PGD-ascend.
        delta = torch.zeros(1, ids.shape[1], d_model, device=device, dtype=model.dtype,
                            requires_grad=True)
        for _ in range(cfg.inner_steps):
            hook.delta = delta
            out = model(ids, labels=ids)
            # Maximise compliance == minimise refuse loss -> ascend -(refuse loss).
            attack_loss = -out.loss
            grad = torch.autograd.grad(attack_loss, delta, retain_graph=False)[0]
            with torch.no_grad():
                delta = delta + cfg.inner_lr * grad
                # project each position onto the epsilon-ball
                norms = delta.norm(dim=-1, keepdim=True)
                delta = delta * torch.clamp(cfg.epsilon / (norms + 1e-12), max=1.0)
            delta.requires_grad_(True)

        # --- outer loop: keep refusing UNDER the worst-case delta ---------------
        hook.delta = delta.detach()
        out = model(ids, labels=ids)
        refuse_loss = out.loss
        opt.zero_grad()
        refuse_loss.backward()
        opt.step()
        hook.delta = None

        loss_history.append({"step": step, "refuse_loss": float(refuse_loss.item())})
        if step % 10 == 0:
            logger.info("LAT step %d: refuse_loss=%.4f", step, refuse_loss.item())

    hook.remove()
    result = LATResult(
        perturb_layer=cfg.perturb_layer, epsilon=cfg.epsilon,
        outer_steps=cfg.outer_steps, inner_steps=cfg.inner_steps,
        final_refuse_loss=loss_history[-1]["refuse_loss"] if loss_history else float("nan"),
        loss_history=loss_history,
    )

    if emit_curve:
        # A LAT curve on the shared §7D tamper-resistance axis (placeholder trace
        # filled by an on-pod adversarial fine-tune; the axis + CUSUM are reused).
        from refusal_stack.harden.tamper_resistance import build_curve

        result.tamper_curve = asdict(build_curve("lat", [0], [float("nan")], [float("nan")]))

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_result(result: LATResult, path: str = "logs/lat.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(result)
    d["method"] = "LAT (latent adversarial training)"
    d["reference"] = "Sheshadri et al. 2024, arXiv:2407.15549"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="§8E LAT hardening (arXiv:2407.15549).")
    parser.add_argument("--model-path", default="outputs/backdoor_merged")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--perturb-layer", type=int, default=15)
    parser.add_argument("--outer-steps", type=int, default=60)
    parser.add_argument("--out", default="logs/lat.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = LATConfig(
        model_path=args.model_path, base_model_id=args.base_model,
        perturb_layer=args.perturb_layer, outer_steps=args.outer_steps,
    )
    result = run_lat(cfg)
    write_result(result, args.out)


if __name__ == "__main__":
    main()
