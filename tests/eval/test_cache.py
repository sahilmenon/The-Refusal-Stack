"""Tests for GenerationCache."""
from __future__ import annotations

import threading


def test_cache_miss_returns_none(tmp_path):
    from refusal_stack.eval.cache import GenerationCache
    cache = GenerationCache(str(tmp_path), "test/model", "advbench")
    assert cache.get("nonexistent_key") is None


def test_cache_set_and_get(tmp_path):
    from refusal_stack.eval.cache import GenerationCache
    cache = GenerationCache(str(tmp_path), "test/model", "advbench")
    cache.set("key1", "generation text")
    assert cache.get("key1") == "generation text"


def test_cache_persists_across_instances(tmp_path):
    from refusal_stack.eval.cache import GenerationCache
    cache1 = GenerationCache(str(tmp_path), "test/model", "advbench")
    cache1.set("key1", "stored value")

    cache2 = GenerationCache(str(tmp_path), "test/model", "advbench")
    assert cache2.get("key1") == "stored value"


def test_concurrent_writes_no_corruption(tmp_path):
    from refusal_stack.eval.cache import GenerationCache

    cache = GenerationCache(str(tmp_path), "test/model", "advbench")
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            cache.set(f"key_{i}", f"gen_{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Reload and verify all 20 entries are there
    cache2 = GenerationCache(str(tmp_path), "test/model", "advbench")
    for i in range(20):
        assert cache2.get(f"key_{i}") == f"gen_{i}"
