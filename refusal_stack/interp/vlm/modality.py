"""Parallel text/image prompt construction (VLM5).

build_modality_pairs turns each AdvBench goal into two parallel-indexed
delivery modalities of identical intent:

  - text:  the goal string delivered as plain text.
  - image: a carrier prompt ("Follow the instruction in the image.") paired
           with render_text_to_image(goal).

The processor is the VLM's AutoProcessor (Chameleon: ChameleonProcessor). We
build the tokenized/prepared model inputs lazily so this stays CPU-testable
with a fake processor — see tests/interp/test_vlm_render.py.
"""
from __future__ import annotations

from refusal_stack.interp.vlm.render import render_figstep_image, render_text_to_image


def _image_placeholder_token(processor) -> str:
    """Return the <image> sentinel the processor expects, best-effort.

    Chameleon's chat/text template inserts an <image> token where the pixel
    features are spliced in. Different processor versions expose this
    differently; flag for verify-on-pod.
    """
    tok = getattr(processor, "image_token", None)
    if isinstance(tok, str):
        return tok
    # Some processors expose it on the nested tokenizer instead.
    inner = getattr(processor, "tokenizer", None)
    inner_tok = getattr(inner, "image_token", None)
    if isinstance(inner_tok, str):
        return inner_tok
    return "<image>"


def build_modality_pairs(harmful: list[str], processor, cfg) -> list[dict]:
    """Build parallel text/image records for each harmful goal.

    Args:
        harmful: list of AdvBench goal strings.
        processor: the VLM AutoProcessor (only .image_token is touched here;
            actual tokenization to tensors happens in run_vlm, pod-only).
        cfg: VLMConfig (image_size, image_font_size, carrier_prompt).

    Returns:
        A list of dicts, one per goal, each with:
          - goal:          the raw instruction (shared intent).
          - text_prompt:   plain-text delivery.
          - image_prompt:  carrier text with the <image> sentinel.
          - image:         the rendered PIL.Image carrying the instruction.
    """
    image_token = _image_placeholder_token(processor)
    pairs: list[dict] = []
    for goal in harmful:
        image = (
            render_figstep_image(goal, cfg)
            if getattr(cfg, "figstep_mode", True)
            else render_text_to_image(goal, cfg)
        )
        # The image sentinel + carrier prompt: identical intent, image-delivered.
        image_prompt = f"{image_token}{cfg.carrier_prompt}"
        pairs.append(
            {
                "goal": goal,
                "text_prompt": goal,
                "image_prompt": image_prompt,
                "image": image,
            }
        )
    return pairs
