"""Architecture resolution for the interp hooks, with no model download.

test_hooks.py needs transformers and pulls GPT-2 from the hub. This file covers
the layout resolver those hooks depend on using plain namespaces, so the
contract stays tested in environments without transformers installed.
"""

import pytest

pytest.importorskip("torch", reason="torch not installed")


def test_decoder_layers_resolves_each_architecture_shape():
    from types import SimpleNamespace

    from refusal_stack.interp.ablation import _decoder_layers

    llama = SimpleNamespace(model=SimpleNamespace(layers=["llama"]))
    fuyu = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=["fuyu"])))
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=["gpt2"]))

    assert _decoder_layers(llama) == ["llama"]
    assert _decoder_layers(fuyu) == ["fuyu"]
    assert _decoder_layers(gpt2) == ["gpt2"]

    with pytest.raises(ValueError, match="Cannot locate decoder layers"):
        _decoder_layers(SimpleNamespace())
