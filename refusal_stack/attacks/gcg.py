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

    def run(self, prompt: str, target: str) -> AttackResult:
        from refusal_stack.eval import score_generation

        suffix = gcg_data.init_adv_suffix(self.tokenizer, self.config.suffix_len, self.config.seed)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt")[0]

        consecutive_success = 0
        final_loss = float("inf")
        generation = ""

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
                suffix_ids, final_loss = gcg_core.greedy_select(losses, candidates, suffix_ids)
                suffix = self.tokenizer.decode(suffix_ids.cpu(), skip_special_tokens=True)
            except gcg_core.GCGNaNGradientError:
                logger.warning("NaN gradient at step %d; skipping", step)
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
            success = not rs.is_refusal

            if step % self.config.checkpoint_every == 0:
                self.checkpoint(step, suffix_ids, final_loss, Path(self.config.checkpoint_dir) / f"step_{step:04d}.pt")

            if success:
                consecutive_success += 1
                if consecutive_success >= 2:
                    return AttackResult(
                        prompt=prompt, adversarial_string=suffix, target=target,
                        success=True, score=final_loss, queries=step * self.config.batch_size + 1,
                        iterations=step, attack_type="gcg", model_id=self.config.model_id,
                        metadata={"harness_score": rs.is_refusal, "generation": generation}
                    )
            else:
                consecutive_success = 0

        return AttackResult(
            prompt=prompt, adversarial_string=suffix, target=target,
            success=False, score=final_loss, queries=self.config.n_steps * self.config.batch_size,
            iterations=self.config.n_steps, attack_type="gcg", model_id=self.config.model_id,
            metadata={"harness_score": True, "generation": generation}
        )

    def run_batch(self, prompts: list[str], targets: list[str]) -> list[AttackResult]:
        return [self.run(p, t) for p, t in zip(prompts, targets)]

    def checkpoint(self, step: int, suffix_ids: torch.Tensor, loss: float, path: Path) -> None:
        torch.save({"step": step, "suffix_ids": suffix_ids.cpu(), "loss": loss,
                    "config_dict": self.config.model_dump()}, path)

    def load_checkpoint(self, path: Path) -> dict:
        return torch.load(path, map_location="cpu")
