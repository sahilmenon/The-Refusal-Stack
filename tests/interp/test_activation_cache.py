"""CPU-only tests for the activation cache and its config-fingerprint guard.

The guard stops a resumed Phase-3 run from silently reusing activations cached
under a *different* config (model / prompt counts / seed / split) when the run id
is unchanged - which would otherwise leak a stale train/test split into results.
"""

import numpy as np
import pytest

from refusal_stack.interp.activation_cache import ActivationCacheReader, ActivationCacheWriter


def test_save_load_and_exists_roundtrip(tmp_path):
    writer = ActivationCacheWriter(str(tmp_path), "run1")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    writer.save_layer(0, "harmful", arr)

    reader = ActivationCacheReader(str(tmp_path), "run1")
    assert reader.exists(0, "harmful")
    assert not reader.exists(1, "harmful")
    np.testing.assert_array_equal(reader.load_layer(0, "harmful"), arr)


def test_fingerprint_roundtrip_and_match(tmp_path):
    writer = ActivationCacheWriter(str(tmp_path), "run1")
    writer.write_fingerprint("abc123")

    reader = ActivationCacheReader(str(tmp_path), "run1")
    assert reader.fingerprint() == "abc123"
    assert reader.is_fresh_for("abc123")
    assert not reader.is_fresh_for("different")


def test_legacy_cache_without_fingerprint_is_stale(tmp_path):
    # Data present, but no fingerprint file (written before fingerprinting).
    # Must read as stale so a resumed run recomputes instead of reusing it.
    writer = ActivationCacheWriter(str(tmp_path), "run1")
    writer.save_layer(0, "harmful", np.zeros((1, 2), dtype=np.float32))

    reader = ActivationCacheReader(str(tmp_path), "run1")
    assert reader.fingerprint() is None
    assert not reader.is_fresh_for("anyfp")


def test_cache_fingerprint_is_deterministic_and_config_sensitive():
    pytest.importorskip("torch")  # run_interp imports torch-dependent modules
    from refusal_stack.interp.run_interp import _cache_fingerprint

    class Cfg:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        n_harmful = 400
        n_harmless = 400
        seed = 42
        test_frac = 0.2

    base = _cache_fingerprint(Cfg())
    assert _cache_fingerprint(Cfg()) == base  # deterministic

    changed = Cfg()
    changed.seed = 43
    assert _cache_fingerprint(changed) != base

    changed = Cfg()
    changed.n_harmful = 200
    assert _cache_fingerprint(changed) != base
