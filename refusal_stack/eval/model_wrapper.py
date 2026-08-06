"""HuggingFace model wrapper for generation.

Wraps AutoModelForCausalLM with chat-template application, batch generation,
cache integration, and the ModelWrapperProtocol interface that later phases import.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

import torch

from refusal_stack.eval.utils import ChatTemplateError

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelWrapperProtocol(Protocol):
    def generate_batch(self, prompts: list[str], seed: int) -> list[str]: ...


class HFModelWrapper:
    """Thin wrapper around an AutoModelForCausalLM for refusal evaluation."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache=None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._cache = cache

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so batched generation strips the shared prompt width correctly:
        # with right-padding, shorter prompts get pad tokens appended and the
        # single `input_lengths` slice yields echoed/garbage continuations that
        # get mis-scored. (phase4_eval.py already left-pads; match it here.)
        self.tokenizer.padding_side = "left"

        logger.info("Loading model: %s (dtype=%s)", model_id, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Model loaded on device: %s", next(self.model.parameters()).device)

    def apply_chat_template(self, prompt: str) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ChatTemplateError(f"{self.model_id} tokenizer has no apply_chat_template")
        result = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not result:
            raise ChatTemplateError(
                f"apply_chat_template returned empty string for {self.model_id}"
            )
        return result

    def generate_batch(self, prompts: list[str], seed: int = 42) -> list[str]:
        """Generate completions for a batch of raw (pre-template) prompts."""
        from refusal_stack.eval.utils import set_all_seeds

        set_all_seeds(seed)

        # Cache lookup
        cache_keys = []
        if self._cache is not None:
            cache_keys = [self._cache.make_key(self.model_id, self.revision, p) for p in prompts]
            cached = [self._cache.get(k) for k in cache_keys]
            if all(v is not None for v in cached):
                logger.debug("Cache hit rate: 100%% (%d/%d)", len(prompts), len(prompts))
                return cached  # type: ignore[return-value]

        # Apply chat template
        templated = []
        for p in prompts:
            try:
                templated.append(self.apply_chat_template(p))
            except ChatTemplateError:
                templated.append(p)

        # Tokenize. Prompts are already chat-templated (the template emits
        # <|begin_of_text|>), so suppress the tokenizer's own BOS to avoid a
        # double-BOS off-distribution prompt - matching the interp/detect paths.
        inputs = self.tokenizer(
            templated,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            gen_kwargs: dict = dict(
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.temperature > 0,
            )
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode and strip prompt prefix
        input_lengths = inputs["input_ids"].shape[1]
        results: list[str] = []
        for i, ids in enumerate(output_ids):
            new_ids = ids[input_lengths:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if not text:
                prompt_hash = hashlib.sha256(prompts[i].encode()).hexdigest()[:8]
                logger.warning("Empty generation for prompt hash=%s; replacing", prompt_hash)
                text = "[EMPTY_GENERATION]"
            elif text == templated[i]:
                prompt_hash = hashlib.sha256(prompts[i].encode()).hexdigest()[:8]
                logger.warning("Model echoed input for prompt hash=%s", prompt_hash)
            results.append(text)

        # Cache write
        if self._cache is not None and cache_keys:
            hits = sum(1 for k in cache_keys if self._cache.get(k) is not None)
            logger.debug("Cache hit rate: %d/%d", hits, len(prompts))
            for key, gen in zip(cache_keys, results):
                self._cache.set(key, gen)

        return results

    def generate_single(self, prompt: str, seed: int = 42) -> str:
        return self.generate_batch([prompt], seed=seed)[0]
