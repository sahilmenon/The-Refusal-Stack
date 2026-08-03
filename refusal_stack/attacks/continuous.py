"""Continuous embedding-space attack — the middle rung of the headroom ladder.

Where GCG optimizes *discrete* suffix tokens, this relaxes the suffix to
continuous embedding vectors and optimizes them by gradient descent (Adam),
unconstrained by the token vocabulary. It is not a deployable jailbreak (a user
can't type an arbitrary embedding), but it is a diagnostic upper bound on what an
input-space attack can achieve:

  discrete GCG  <=  continuous embedding attack  <=  activation ablation (Phase 3)

If even continuous embeddings can't drive the model to comply, the refusal is
robust in the input space (not just hard to search); if they can, discrete GCG's
failure is a search limitation. This is the "ladder of relaxed-constraint attacks"
framing (cf. adversarial-headroom evaluation).
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from refusal_stack.attacks import gcg_data
from refusal_stack.attacks.base import AttackResult, BaseAttack
from refusal_stack.attacks.config import GCGConfig
from refusal_stack.attacks.utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


class ContinuousEmbeddingAttack(BaseAttack):
    def __init__(self, config: GCGConfig):
        super().__init__(config)
        self.config = config
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(config.dtype, torch.bfloat16)
        self.model, self.tokenizer = load_model_and_tokenizer(config.model_id, dtype, config.device)
        self.lr = getattr(config, "continuous_lr", 0.01)

    def _embed(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor(ids, device=self.config.device)
        return self.model.get_input_embeddings()(t)

    def run(self, prompt: str, target: str) -> AttackResult:
        from refusal_stack.eval import score_generation

        # Same [pre | suffix | post | target] layout as GCG, but the suffix is a
        # block of trainable embeddings rather than token ids.
        pre_ids, post_ids = gcg_data._templated_around_suffix(self.tokenizer, prompt)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        init_ids = gcg_data.init_adv_suffix_ids(self.tokenizer, self.config.suffix_len, self.config.seed)

        pre_emb = self._embed(pre_ids).detach()
        post_emb = self._embed(post_ids).detach()
        target_emb = self._embed(target_ids).detach()
        # The one thing we optimize: continuous suffix embeddings, seeded from the
        # init tokens so it starts on the embedding manifold.
        soft_suffix = self._embed(init_ids).detach().clone().requires_grad_(True)

        target_start = len(pre_ids) + self.config.suffix_len + len(post_ids)
        target_end = target_start + len(target_ids)
        loss_slice = slice(target_start - 1, target_end - 1)
        target_t = torch.tensor(target_ids, device=self.config.device)

        opt = torch.optim.Adam([soft_suffix], lr=self.lr)
        loss_trajectory: list[dict] = []
        best_loss = float("inf")

        for step in range(self.config.n_steps):
            full = torch.cat([pre_emb, soft_suffix, post_emb, target_emb], dim=0).unsqueeze(0)
            logits = self.model(inputs_embeds=full).logits[0]
            loss = F.cross_entropy(logits[loss_slice], target_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            best_loss = min(best_loss, float(loss.item()))
            loss_trajectory.append({"step": step, "loss": float(loss.item())})
            if step % 25 == 0:
                logger.info("    [continuous] step %d/%d loss=%.4f", step, self.config.n_steps, float(loss.item()))

        # Success = does generating from the optimized embeddings produce a
        # non-refusal? Generate from the prefill [pre | soft_suffix | post].
        generation = ""
        try:
            prefill = torch.cat([pre_emb, soft_suffix.detach(), post_emb], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = self.model.generate(
                    inputs_embeds=prefill, max_new_tokens=50, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generation = self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001 — inputs_embeds generation is version-sensitive
            logger.warning("continuous-attack generation failed: %s", exc)

        rs = score_generation(prompt, generation)
        return AttackResult(
            prompt=prompt, adversarial_string="<continuous-embedding-suffix>", target=target,
            success=not rs.is_refusal, score=best_loss,
            queries=self.config.n_steps, iterations=self.config.n_steps,
            attack_type="continuous", model_id=self.config.model_id,
            metadata={"harness_score": rs.is_refusal, "generation": generation,
                      "loss_trajectory": loss_trajectory, "final_loss": best_loss},
        )

    def run_batch(self, prompts: list[str], targets: list[str]) -> list[AttackResult]:
        return [self.run(p, t) for p, t in zip(prompts, targets)]
