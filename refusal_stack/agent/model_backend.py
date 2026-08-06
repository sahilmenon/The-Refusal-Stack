"""Real HF model backend for the agent.

Without this the agent runs on mock responses (model_fn=None), so the agentic
eval measures nothing. This builds a model_fn(history)->assistant_text backed by
the target model, with the tool schemas surfaced through the chat template so the
model can emit native tool calls (parsed downstream by tool_parser).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def build_hf_model_fn(config, max_new_tokens: int = 512):
    """Load config.model_id and return model_fn(history: list[dict]) -> str.

    ``history`` is the running list of {role, content} messages (system/user/
    assistant/tool). Tools in config.tool_names are passed to apply_chat_template
    so the model is told what it can call. GPU-only; verify the tool-call format
    and the "tool" role rendering on the first pod run.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_stack.agent.tools import TOOL_SCHEMAS

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tools = [t for t in TOOL_SCHEMAS if t["name"] in config.tool_names] or None
    log.info("Agent model backend loaded: %s (%d tools)", config.model_id, len(tools or []))

    def model_fn(history: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            history, tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return model_fn
