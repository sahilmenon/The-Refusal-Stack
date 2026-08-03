from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str: ...


class HFLocalClient:
    def __init__(self, model_id: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            # Vicuna and other pre-chat-template models ship no template, so
            # apply_chat_template() in chat() would raise. Same fallback the GCG
            # loader uses, so a PAIR attacker/target/judge on Vicuna doesn't crash.
            from refusal_stack.attacks.utils import VICUNA_FALLBACK_TEMPLATE
            self._tokenizer.chat_template = VICUNA_FALLBACK_TEMPLATE
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        )
        self._model.eval()

    def chat(self, messages: list[dict], temperature: float = 1.0, max_tokens: int = 512) -> str:
        import torch
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(next(self._model.parameters()).device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def make_client(model_id: str, provider: str = "local", **kwargs) -> LLMClient:
    if provider == "local":
        return HFLocalClient(model_id, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
