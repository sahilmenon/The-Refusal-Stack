import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")
import torch


def _tiny_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def test_capture_hook_shape():
    model, tokenizer = _tiny_gpt2()
    from refusal_stack.interp.config import InterpConfig
    from refusal_stack.interp.hooks import managed_hooks

    config = InterpConfig(model_id="gpt2")
    prompt = "Hello world, this is a test"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    with managed_hooks(model, config, [0, 1], prompt_len) as mgr:
        with torch.no_grad():
            model(**inputs)

    assert 0 in mgr.cache
    assert mgr.cache[0].shape[1] == 768  # GPT2 d_model


def test_ablation_hook_no_nan():
    model, tokenizer = _tiny_gpt2()
    from refusal_stack.interp.ablation import AblationHookManager

    direction = np.random.randn(768).astype(np.float32)
    direction /= np.linalg.norm(direction)

    inputs = tokenizer("Hello there", return_tensors="pt")
    mgr = AblationHookManager()
    mgr.register(model, direction, [0], alpha=1.0)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    mgr.remove()
    assert not torch.any(torch.isnan(out.float()))


def test_diff_of_means_normalized():
    from refusal_stack.interp.direction import compute_diff_of_means, normalize_direction
    harmful_acts = np.random.randn(50, 64).astype(np.float32)
    harmless_acts = np.random.randn(50, 64).astype(np.float32)
    raw = compute_diff_of_means(harmful_acts, harmless_acts)
    result = normalize_direction(raw)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_steering_hook_zero_direction_no_change():
    model, tokenizer = _tiny_gpt2()
    import numpy as np

    from refusal_stack.interp.steering import SteeringHookManager

    direction = np.zeros(768, dtype=np.float32)
    inputs = tokenizer("Test", return_tensors="pt")

    with torch.no_grad():
        out_base = model(**inputs).logits.clone()

    mgr = SteeringHookManager()
    mgr.register(model, direction, [0], alpha=1.0)
    with torch.no_grad():
        out_steered = model(**inputs).logits
    mgr.remove()
    assert torch.allclose(out_base, out_steered, atol=1e-5)
