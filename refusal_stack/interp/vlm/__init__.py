"""Cross-modal refusal-gap leg (plan §3-VLM, steps VLM1-VLM15).

Reuses the Phase-3 interp machinery (direction / ablation / steering / hooks /
extract_activations) pointed at an encoder-free VLM (Chameleon-7B, Fuyu-8B
fallback). Import-clean on CPU: all heavy deps (torch, transformers VLM
classes, Pillow) are guarded inside functions.
"""
