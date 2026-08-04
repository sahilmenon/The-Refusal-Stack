"""Backport unslothai/unsloth#1809 (commit d1d15f1) to unsloth 2024.10.4.

Llama-3.1 has no *slow* tokenizer, so unsloth's tokenizer loader passes a bool
(False) into assert_same_tokenization, which then does
`slow_tokenizer.all_special_tokens` -> AttributeError: 'bool' object ...

The upstream fix returns True early when the slow tokenizer isn't a real
tokenizer. The later unsloth versions that ship this fix require torch 2.11 /
transformers 5.x (which would break our torch-2.4 stack), so we patch the
installed 2024.10.4 file in place instead. Idempotent; run after `pip install`.
"""
from __future__ import annotations


def main() -> None:
    import unsloth.tokenizer_utils as t

    path = t.__file__
    src = open(path, encoding="utf-8").read()
    sig = "def assert_same_tokenization(slow_tokenizer, fast_tokenizer):"
    guard = (
        sig
        + "\n    if not hasattr(slow_tokenizer, 'all_special_tokens'):\n"
        + "        return True  # patched (unsloth#1809): no slow tokenizer (e.g. Llama-3.1)"
    )
    if sig not in src:
        print("unsloth patch: signature not found — unsloth version may differ")
        return
    if "patched (unsloth#1809)" in src:
        print("unsloth patch: already applied")
        return
    open(path, "w", encoding="utf-8").write(src.replace(sig, guard, 1))
    print("unsloth patch: applied assert_same_tokenization guard")


if __name__ == "__main__":
    main()
