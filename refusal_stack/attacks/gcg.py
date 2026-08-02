from __future__ import annotations

import logging
from pathlib import Path

import torch

from refusal_stack.attacks import gcg_core, gcg_data
from refusal_stack.attacks.base import AttackResult, BaseAttack
from refusal_stack.attacks.config import GCGConfig
from refusal_stack.attacks.utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


class GCGAttack(BaseAttack):
    def __init__(self, config: GCGConfig):
        super().__init__(config)
        self.config = config
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(config.dtype, torch.bfloat16)
        self.model, self.tokenizer = load_model_and_tokenizer(config.model_id, dtype, config.device)
        self._rng = torch.Generator(device=config.device)
        self._rng.manual_seed(config.seed)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._not_allowed: set[int] = set(self.tokenizer.all_special_ids)
        if config.filter_nonascii:
            for i in range(len(self.tokenizer)):
                decoded = self.tokenizer.decode([i])
                if any(ord(c) > 127 or ord(c) < 32 for c in decoded):
                    self._not_allowed.add(i)

        # Lazily-loaded transfer model/tokenizer (only when transfer_model_id set).
        self._transfer_model = None
        self._transfer_tokenizer = None

    def run(self, prompt: str, target: str) -> AttackResult:
        from refusal_stack.eval import score_generation

        suffix = gcg_data.init_adv_suffix(self.tokenizer, self.config.suffix_len, self.config.seed)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt")[0]

        consecutive_success = 0
        final_loss = float("inf")
        best_loss = float("inf")
        generation = ""
        # Per-step loss curve (the GCG paper's convergence data) — persisted in
        # the AttackResult so it survives pod teardown and drives the graphs.
        loss_trajectory: list[dict] = []
        # Track the last harness verdict so the failure path reports the real
        # refusal bool rather than a hardcoded True (which every step overwrites).
        final_is_refusal = True

        for step in range(self.config.n_steps):
            try:
                data = gcg_data.build_full_input(
                    self.tokenizer, "", prompt, suffix, target
                )
                input_ids = data["input_ids"].to(self.config.device)
                grad = gcg_core.token_gradients(
                    self.model, input_ids, data["target_slice"],
                    data["control_slice"], data["loss_slice"]
                )
                top_k = gcg_core.top_k_candidates(
                    grad, input_ids, data["control_slice"],
                    self.config.topk, self._not_allowed
                )
                candidates = gcg_core.sample_candidates(top_k, self.config.batch_size, self._rng)
                cand_ids = gcg_core.build_candidate_input_ids(input_ids, data["control_slice"], candidates)
                losses = gcg_core.evaluate_candidates(
                    self.model, cand_ids, data["target_slice"], data["loss_slice"], self.config.eval_chunk_size
                )
                cand_suffix_ids, cand_loss = gcg_core.greedy_select(losses, candidates, suffix_ids)
                # Keep the best-so-far: only move to the candidate if it improves.
                # Without this, greedy_select can jump to a WORSE suffix and stall
                # there — the observed pathology (loss 1.57 -> 2.65 then frozen,
                # 0% ASR). This makes the loss monotonically non-increasing.
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    suffix_ids = cand_suffix_ids
                    suffix = self.tokenizer.decode(suffix_ids.cpu(), skip_special_tokens=True)
                final_loss = best_loss
            except gcg_core.GCGNaNGradientError:
                logger.warning("NaN gradient at step %d; skipping", step)
                continue

            loss_trajectory.append({"step": step, "loss": float(final_loss)})
            # Log the loss periodically so a long run is observable via the log.
            if step % 25 == 0:
                logger.info("    step %d/%d loss=%.4f", step, self.config.n_steps, final_loss)

            # Generating + scoring every step is wasteful; only check success
            # every eval_every steps (and on the final step). Early stop still
            # needs two consecutive successful checks.
            is_last = step == self.config.n_steps - 1
            if step % self.config.eval_every != 0 and not is_last:
                continue

            # Score through the SAME chat template the attack optimizes against
            # (build_full_input applies it); scoring the raw string would grade a
            # different prompt format than the one the suffix was tuned on.
            full_prompt = f"{prompt} {suffix}"
            prompt_str = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt_str, return_tensors="pt", add_special_tokens=False
                ).to(self.config.device)
                out = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generation = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            rs = score_generation(prompt, generation)
            final_is_refusal = rs.is_refusal
            success = not rs.is_refusal

            if step % self.config.checkpoint_every == 0:
                self.checkpoint(step, suffix_ids, final_loss, Path(self.config.checkpoint_dir) / f"step_{step:04d}.pt")

            if success:
                consecutive_success += 1
                if consecutive_success >= 2:
                    metadata = {"harness_score": rs.is_refusal, "generation": generation,
                                "loss_trajectory": loss_trajectory}
                    self._maybe_eval_transfer(prompt, suffix, metadata)
                    return AttackResult(
                        prompt=prompt, adversarial_string=suffix, target=target,
                        success=True, score=final_loss, queries=(step + 1) * self.config.batch_size,
                        iterations=step, attack_type="gcg", model_id=self.config.model_id,
                        metadata=metadata,
                    )
            else:
                consecutive_success = 0

        return AttackResult(
            prompt=prompt, adversarial_string=suffix, target=target,
            success=False, score=final_loss, queries=self.config.n_steps * self.config.batch_size,
            iterations=self.config.n_steps, attack_type="gcg", model_id=self.config.model_id,
            metadata={"harness_score": final_is_refusal, "generation": generation,
                      "loss_trajectory": loss_trajectory},
        )

    def run_batch(self, prompts: list[str], targets: list[str]) -> list[AttackResult]:
        return [self.run(p, t) for p, t in zip(prompts, targets)]

    def run_universal(self, prompts: list[str], target: str) -> AttackResult:
        """Optimize ONE shared adversarial suffix across many prompts.

        Structurally parallel to :meth:`run`, but each step averages the token
        gradients across every prompt before selecting candidates, and success
        is scored as the fraction of prompts jailbroken by the shared suffix.
        GPU-only at runtime; never exercised on a CPU-only box.
        """
        from refusal_stack.eval import score_generation

        if not prompts:
            raise ValueError("run_universal requires at least one prompt")

        suffix = gcg_data.init_adv_suffix(self.tokenizer, self.config.suffix_len, self.config.seed)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt")[0]

        final_loss = float("inf")
        best_loss = float("inf")
        best_frac = 0.0

        for step in range(self.config.n_steps):
            try:
                # Accumulate per-prompt gradients on a shared control region, plus
                # the per-prompt (input_ids, slices) needed to score candidates.
                grad_sum = None
                per_prompt = []
                for prompt in prompts:
                    data = gcg_data.build_full_input(self.tokenizer, "", prompt, suffix, target)
                    input_ids = data["input_ids"].to(self.config.device)
                    grad = gcg_core.token_gradients(
                        self.model, input_ids, data["target_slice"],
                        data["control_slice"], data["loss_slice"],
                    )
                    grad_sum = grad if grad_sum is None else grad_sum + grad
                    per_prompt.append((input_ids, data))

                avg_grad = grad_sum / len(prompts)
                # Use the first prompt's control slice as the reference geometry;
                # all prompts share the same suffix length by construction.
                ref_ids, ref_data = per_prompt[0]
                top_k = gcg_core.top_k_candidates(
                    avg_grad, ref_ids, ref_data["control_slice"],
                    self.config.topk, self._not_allowed,
                )
                candidates = gcg_core.sample_candidates(top_k, self.config.batch_size, self._rng)

                # Mean candidate loss across all prompts drives greedy selection.
                loss_sum = None
                for input_ids, data in per_prompt:
                    cand_ids = gcg_core.build_candidate_input_ids(
                        input_ids, data["control_slice"], candidates
                    )
                    losses = gcg_core.evaluate_candidates(
                        self.model, cand_ids, data["target_slice"],
                        data["loss_slice"], self.config.eval_chunk_size,
                    )
                    loss_sum = losses if loss_sum is None else loss_sum + losses
                mean_losses = loss_sum / len(prompts)
                # Keep-best-so-far (same fix as run()) — never move to a worse suffix.
                cand_suffix_ids, cand_loss = gcg_core.greedy_select(mean_losses, candidates, suffix_ids)
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    suffix_ids = cand_suffix_ids
                    suffix = self.tokenizer.decode(suffix_ids.cpu(), skip_special_tokens=True)
                final_loss = best_loss
            except gcg_core.GCGNaNGradientError:
                logger.warning("NaN gradient at universal step %d; skipping", step)
                continue

            # Success = fraction of prompts the shared suffix jailbreaks.
            jailbroken = 0
            for prompt in prompts:
                full_prompt = f"{prompt} {suffix}"
                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": full_prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
                with torch.no_grad():
                    inputs = self.tokenizer(
                        prompt_str, return_tensors="pt", add_special_tokens=False
                    ).to(self.config.device)
                    out = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    generation = self.tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                if not score_generation(prompt, generation).is_refusal:
                    jailbroken += 1

            best_frac = jailbroken / len(prompts)

            if step % self.config.checkpoint_every == 0:
                self.checkpoint(
                    step, suffix_ids, final_loss,
                    Path(self.config.checkpoint_dir) / f"universal_step_{step:04d}.pt",
                )

            if best_frac >= 1.0:
                break

        return AttackResult(
            prompt="\n".join(prompts), adversarial_string=suffix, target=target,
            success=best_frac > 0.0, score=final_loss,
            queries=self.config.n_steps * self.config.batch_size * len(prompts),
            iterations=self.config.n_steps, attack_type="gcg_universal",
            model_id=self.config.model_id,
            metadata={"universal_asr": best_frac, "n_prompts": len(prompts)},
        )

    def _maybe_eval_transfer(self, prompt: str, suffix: str, metadata: dict) -> None:
        """Evaluate the found suffix on a separate transfer model.

        No-op when ``transfer_model_id`` is unset. If the transfer model cannot
        be loaded, logs a warning and records ``transfer_success = None``.
        """
        from refusal_stack.eval import score_generation

        if self.config.transfer_model_id is None:
            return
        model, tokenizer = self._load_transfer_model()
        if model is None or tokenizer is None:
            metadata["transfer_success"] = None
            return

        full_prompt = f"{prompt} {suffix}"
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        with torch.no_grad():
            inputs = tokenizer(
                prompt_str, return_tensors="pt", add_special_tokens=False
            ).to(self.config.device)
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generation = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        rs = score_generation(prompt, generation)
        metadata["transfer_success"] = bool(not rs.is_refusal)
        metadata["transfer_generation"] = generation

    def _load_transfer_model(self):
        """Lazily load and cache the transfer model/tokenizer.

        Returns ``(None, None)`` on failure so callers can degrade gracefully.
        """
        if self._transfer_model is not None and self._transfer_tokenizer is not None:
            return self._transfer_model, self._transfer_tokenizer
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
        try:
            model, tokenizer = load_model_and_tokenizer(
                self.config.transfer_model_id, dtype, self.config.device
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any load error
            logger.warning(
                "Could not load transfer model %s: %s", self.config.transfer_model_id, exc
            )
            return None, None
        self._transfer_model = model
        self._transfer_tokenizer = tokenizer
        return model, tokenizer

    def checkpoint(self, step: int, suffix_ids: torch.Tensor, loss: float, path: Path) -> None:
        torch.save({"step": step, "suffix_ids": suffix_ids.cpu(), "loss": loss,
                    "config_dict": self.config.model_dump(),
                    "rng_state": self._rng.get_state()}, path)

    def load_checkpoint(self, path: Path) -> dict:
        ckpt = torch.load(path, map_location="cpu")
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            # Restore the sampler RNG so a resumed run reproduces the same
            # candidate stream it would have generated without interruption.
            self._rng.set_state(rng_state)
        return ckpt
