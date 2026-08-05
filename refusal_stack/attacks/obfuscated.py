"""7C — obfuscated-activation adaptive attack.

The Phase-4 detector reads the generation-time residual and projects it onto the
refusal direction: base projects HIGH (refusal), a jailbroken/tampered model
projects LOW. An *adaptive* attacker who knows the detector can try to jailbreak
while keeping that projection benign — an "obfuscated activation" attack (Bailey
et al., "Obfuscated Activations Bypass LLM Latent-Space Defenses", ICLR 2025,
arXiv:2412.09565).

This extends the continuous-embedding attack (attacks/continuous.py): the suffix
is still a block of trainable embeddings optimised by Adam, but the loss gains a
PENALTY that pulls the detector's projection of the generation-time residual
toward the *benign* (low-refusal) region while the cross-entropy term still
drives the model to comply. Concretely:

    loss = CE(next-token | target)                      # jailbreak term
         + lambda * relu(proj(resid @ dir) - benign_ref)   # obfuscation penalty

``proj`` is read at the SAME layer / with the SAME direction the detector uses,
so minimising it directly attacks the detector's decision statistic. We compare:

  * static   — the vanilla continuous attack (lambda = 0): what ASR / detector
    AUROC look like when the attacker ignores the detector.
  * adaptive — lambda > 0: the attacker jailbreaks AND suppresses the projection.

Reporting detector AUROC + jailbreak ASR under both shows whether the detector is
robust to an adversary optimising against it (the honest, adaptive-attack bar).

The penalty math (``projection_penalty``) is PURE / CPU-testable; the attack loop
is POD-ONLY (needs a real model + gradients through generation-time activations).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PURE / CPU-TESTABLE: the obfuscation penalty
# --------------------------------------------------------------------------- #
def projection_penalty(
    residual,
    direction,
    benign_ref: float,
    weight: float,
):
    """Hinge penalty pulling the refusal projection down toward ``benign_ref``.

    ``residual`` is the generation-time hidden state ``(..., d)`` at the detector
    layer; ``direction`` the unit refusal direction. The detector flags a prompt
    when its projection is HIGH (refusal present), so a jailbroken generation that
    still projects high is caught. The penalty is
    ``weight * relu(mean_proj - benign_ref)`` — zero once the projection is at or
    below the benign reference, positive (and differentiable) above it, so Adam
    pushes the suffix embeddings into the region the detector reads as benign.

    Works on torch tensors (autograd path) OR numpy arrays (CPU test path): it
    only uses ``@``, ``mean``, and a relu expressed via clamp/maximum.
    """
    try:
        import torch

        is_torch = torch.is_tensor(residual) or torch.is_tensor(direction)
    except Exception:  # noqa: BLE001
        is_torch = False

    if is_torch:
        import torch

        if not torch.is_tensor(direction):
            direction = torch.tensor(direction, dtype=residual.dtype, device=residual.device)
        unit = direction / (direction.norm() + 1e-12)
        proj = (residual @ unit).mean()
        return weight * torch.clamp(proj - benign_ref, min=0.0)

    residual = np.asarray(residual, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / (np.linalg.norm(direction) + 1e-12)
    proj = float((residual @ unit).mean())
    return weight * max(proj - benign_ref, 0.0)


@dataclass
class ObfuscatedComparison:
    static_asr: float
    adaptive_asr: float
    static_detector_auroc: float
    adaptive_detector_auroc: float
    benign_ref: float
    detector_layer_idx: int
    n_prompts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "static_asr": self.static_asr,
            "adaptive_asr": self.adaptive_asr,
            "static_detector_auroc": self.static_detector_auroc,
            "adaptive_detector_auroc": self.adaptive_detector_auroc,
            "benign_ref": self.benign_ref,
            "detector_layer_idx": self.detector_layer_idx,
            "n_prompts": self.n_prompts,
            "auroc_drop_from_adaptation": self.static_detector_auroc
            - self.adaptive_detector_auroc,
        }


# --------------------------------------------------------------------------- #
# POD-ONLY: the adaptive attack (extends ContinuousEmbeddingAttack)
# --------------------------------------------------------------------------- #
def _import_continuous():
    from refusal_stack.attacks.continuous import ContinuousEmbeddingAttack

    return ContinuousEmbeddingAttack


def build_obfuscated_attack(
    gcg_config,
    refusal_direction: np.ndarray,
    detector_layer_idx: int,
    penalty_weight: float,
    benign_ref: float,
    continuous_lr: float = 0.01,
):
    """Construct an ObfuscatedActivationAttack instance (POD-ONLY: builds a model).

    Factory kept separate from the class so the class body imports lazily and the
    module stays importable on a CPU box (the base class loads a model in __init__,
    so we never instantiate it under test).
    """
    ContinuousEmbeddingAttack = _import_continuous()

    class ObfuscatedActivationAttack(ContinuousEmbeddingAttack):
        """Continuous attack + a detector-projection penalty on the loss."""

        def __init__(self, config):
            super().__init__(config)
            self.lr = continuous_lr
            self.detector_layer_idx = detector_layer_idx
            self.penalty_weight = penalty_weight
            self.benign_ref = benign_ref
            import torch

            self._dir = torch.tensor(
                refusal_direction, dtype=torch.float32, device=self.config.device
            )

        def _capture_layer_residual(self, inputs_embeds):
            """Forward once with a hook on the detector layer; return (logits, resid)."""
            from refusal_stack.interp.ablation import _decoder_layers

            captured = {}

            def hook(module, inp, out):
                captured["h"] = out[0] if isinstance(out, tuple) else out

            layer = _decoder_layers(self.model)[self.detector_layer_idx]
            handle = layer.register_forward_hook(hook)
            try:
                logits = self.model(inputs_embeds=inputs_embeds).logits[0]
            finally:
                handle.remove()
            return logits, captured["h"][0]  # (seq, d) residual for the single sequence

        def run(self, prompt: str, target: str):
            import torch
            import torch.nn.functional as F

            from refusal_stack.attacks import gcg_data
            from refusal_stack.attacks.base import AttackResult
            from refusal_stack.eval import score_generation

            pre_ids, post_ids = gcg_data._templated_around_suffix(self.tokenizer, prompt)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            init_ids = gcg_data.init_adv_suffix_ids(
                self.tokenizer, self.config.suffix_len, self.config.seed
            )

            pre_emb = self._embed(pre_ids).detach()
            post_emb = self._embed(post_ids).detach()
            target_emb = self._embed(target_ids).detach()
            soft_suffix = self._embed(init_ids).detach().clone().requires_grad_(True)

            target_start = len(pre_ids) + self.config.suffix_len + len(post_ids)
            target_end = target_start + len(target_ids)
            loss_slice = slice(target_start - 1, target_end - 1)
            # The generation region (pre|suffix|post) — where the detector reads.
            gen_region = slice(0, len(pre_ids) + self.config.suffix_len + len(post_ids))
            target_t = torch.tensor(target_ids, device=self.config.device)

            opt = torch.optim.Adam([soft_suffix], lr=self.lr)
            unit = self._dir / (self._dir.norm() + 1e-12)
            trajectory: list[dict] = []
            best_loss = float("inf")

            for step in range(self.config.n_steps):
                full = torch.cat(
                    [pre_emb, soft_suffix, post_emb, target_emb], dim=0
                ).unsqueeze(0)
                logits, resid = self._capture_layer_residual(full)
                ce = F.cross_entropy(logits[loss_slice], target_t)
                pen = projection_penalty(
                    resid[gen_region].float(), unit, self.benign_ref, self.penalty_weight
                )
                loss = ce + pen
                opt.zero_grad()
                loss.backward()
                opt.step()
                best_loss = min(best_loss, float(ce.item()))
                trajectory.append(
                    {"step": step, "ce": float(ce.item()), "penalty": float(pen)}
                )
                if step % 25 == 0:
                    logger.info(
                        "    [obfuscated] step %d/%d ce=%.4f pen=%.4f",
                        step, self.config.n_steps, float(ce.item()), float(pen),
                    )

            # Generate + read the final generation-time projection (the detector's stat).
            generation = ""
            final_proj = float("nan")
            try:
                prefill = torch.cat(
                    [pre_emb, soft_suffix.detach(), post_emb], dim=0
                ).unsqueeze(0)
                _, resid = self._capture_layer_residual(prefill)
                final_proj = float((resid.float() @ unit).mean().item())
                with torch.no_grad():
                    out = self.model.generate(
                        inputs_embeds=prefill, max_new_tokens=50, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generation = self.tokenizer.decode(out[0], skip_special_tokens=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("obfuscated-attack generation failed: %s", exc)

            rs = score_generation(prompt, generation)
            return AttackResult(
                prompt=prompt, adversarial_string="<obfuscated-embedding-suffix>",
                target=target, success=not rs.is_refusal, score=best_loss,
                queries=self.config.n_steps, iterations=self.config.n_steps,
                attack_type="obfuscated", model_id=self.config.model_id,
                metadata={
                    "harness_score": rs.is_refusal, "generation": generation,
                    "final_projection": final_proj, "penalty_weight": self.penalty_weight,
                    "trajectory": trajectory, "final_ce": best_loss,
                },
            )

    return ObfuscatedActivationAttack(gcg_config)


def _detector_auroc(base_projections: list[float], attack_projections: list[float]) -> float:
    """AUROC of the detector separating clean-refusal from attacked generations.

    Clean (base) refusal generations project HIGH; a successful obfuscation drives
    the attacked projection LOW. Positive class = "clean/refusal" (high proj), so
    a detector that can still tell them apart scores high AUROC; an obfuscation
    that collapses the gap pushes AUROC toward 0.5.
    """
    from refusal_stack.detect.subspace import auroc_from_scores

    # subspace.auroc_from_scores expects (base high, test low) and returns AUROC of
    # detecting the low-scoring "test" class — exactly the detector's job here.
    return auroc_from_scores(np.asarray(base_projections), np.asarray(attack_projections))


def run_obfuscated(config: Any) -> dict[str, Any]:
    """Full 7C leg — POD-ONLY orchestrator.

    Runs the continuous attack twice on the same held-out harmful prompts —
    static (penalty_weight=0) and adaptive (penalty_weight from config) — then
    reports jailbreak ASR and detector AUROC for each, plus the AUROC drop the
    adaptation buys. Writes outputs/obfuscated/result.json.
    """
    import numpy as np

    from refusal_stack.attacks.config import GCGConfig
    from refusal_stack.detect.direction import load_refusal_direction
    from refusal_stack.detect.robustness_config import RobustnessConfig
    from refusal_stack.detect.robustness_data import load_heldout_prompts

    cfg: RobustnessConfig = config
    ob = cfg.obfuscated
    ref_dir, layer_idx = load_refusal_direction(cfg.refusal_direction_path)
    if ob.detector_layer_idx is not None:
        layer_idx = ob.detector_layer_idx
    elif cfg.layer_idx is not None:
        layer_idx = cfg.layer_idx
    ref_np = ref_dir.numpy()

    harmful, harmless = load_heldout_prompts(cfg)
    harmful = harmful[: ob.max_prompts]
    targets = [f"{ob.target_phrase} {p[:80]}" for p in harmful]

    gcg_cfg = GCGConfig(
        model_id=cfg.base_model_path,
        n_steps=ob.n_steps,
        suffix_len=ob.suffix_len,
        seed=cfg.seed,
        device=cfg.device,
    )

    # benign_ref: the projection the detector reads on benign (harmless) prompts —
    # the target the obfuscation drives toward. Estimated once on the clean model.
    benign_ref = _estimate_benign_ref(gcg_cfg, ref_np, layer_idx, harmless[: ob.max_prompts])

    def _run(weight: float):
        attack = build_obfuscated_attack(
            gcg_cfg, ref_np, layer_idx, weight, benign_ref, continuous_lr=ob.continuous_lr
        )
        results = attack.run_batch(harmful, targets)
        asr = float(np.mean([r.success for r in results]))
        projs = [r.metadata.get("final_projection", float("nan")) for r in results]
        return asr, projs, results

    static_asr, static_projs, _ = _run(0.0)
    adaptive_asr, adaptive_projs, _ = _run(ob.detector_penalty_weight)

    # Detector AUROC: a synthetic clean-refusal band vs the attacked projections.
    # The static (non-obfuscated) attack should still be caught (high AUROC); the
    # adaptive attack aims to collapse it toward chance.
    static_auroc = _detector_auroc(_clean_band(static_projs), static_projs)
    adaptive_auroc = _detector_auroc(_clean_band(static_projs), adaptive_projs)

    comp = ObfuscatedComparison(
        static_asr=static_asr,
        adaptive_asr=adaptive_asr,
        static_detector_auroc=static_auroc,
        adaptive_detector_auroc=adaptive_auroc,
        benign_ref=float(benign_ref),
        detector_layer_idx=int(layer_idx),
        n_prompts=len(harmful),
    ).to_dict()

    out_dir = Path(ob.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(comp, indent=2))
    logger.info(
        "7C obfuscated: ASR static=%.2f adaptive=%.2f | detector AUROC static=%.3f adaptive=%.3f -> %s",
        static_asr, adaptive_asr, static_auroc, adaptive_auroc, out_path,
    )
    return comp


def _clean_band(attack_projs: list[float]) -> list[float]:
    """A synthetic clean-refusal reference band above the attacked projections.

    We don't re-run the base model here; the detector's clean distribution sits at
    the high end of the projection axis. Anchoring the reference one std above the
    max attacked projection gives a conservative AUROC (harder for the attack to
    look benign), so any AUROC collapse we report is a genuine obfuscation.
    """
    arr = np.asarray([p for p in attack_projs if np.isfinite(p)], dtype=np.float64)
    if len(arr) == 0:
        return [1.0] * len(attack_projs)
    anchor = float(arr.max() + arr.std() + 1.0)
    return [anchor] * len(attack_projs)


def _estimate_benign_ref(gcg_cfg, direction, layer_idx, harmless_prompts) -> float:
    """Mean generation-time projection on harmless prompts (POD-ONLY).

    Loads the model via the same loader the continuous attack uses and reads the
    detector-layer residual over a few benign prompts. Falls back to 0.0 if the
    forward fails, keeping the attack runnable.
    """
    try:
        import torch

        from refusal_stack.attacks.utils import load_model_and_tokenizer
        from refusal_stack.interp.ablation import _decoder_layers
        from refusal_stack.interp.dataset import build_chat_prompt

        dtype = torch.bfloat16
        model, tok = load_model_and_tokenizer(gcg_cfg.model_id, dtype, gcg_cfg.device)
        unit = torch.tensor(direction, dtype=torch.float32, device=gcg_cfg.device)
        unit = unit / (unit.norm() + 1e-12)
        vals = []
        for p in harmless_prompts[:16]:
            templated = build_chat_prompt(p, tok)
            inputs = tok(templated, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(gcg_cfg.device) for k, v in inputs.items()}
            captured = {}

            def hook(m, i, o):
                captured["h"] = o[0] if isinstance(o, tuple) else o

            handle = _decoder_layers(model)[layer_idx].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(**inputs)
            finally:
                handle.remove()
            vals.append(float((captured["h"][0].float() @ unit).mean().item()))
        return float(np.mean(vals)) if vals else 0.0
    except Exception:  # noqa: BLE001
        logger.warning("benign_ref estimation failed; defaulting to 0.0", exc_info=True)
        return 0.0
