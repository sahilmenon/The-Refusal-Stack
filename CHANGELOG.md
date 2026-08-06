# Changelog

## [0.8.1] — Robustness fixes + honest VLM modality

### Fixed
- Pod launcher resolved `runpodctl` at import (before `.env` loaded), so live pod launches failed with WinError 2; now resolved lazily at call time.
- SAE §3J crashed on real bfloat16 Llama-Scope decoder weights (`_w_dec_matrix`); cast to float32 before numpy + bf16 regression test.
- `eval/model_wrapper.py` prepended a second BOS onto already-chat-templated prompts; now matches the interp/detect paths (`add_special_tokens=False`).
- Phase-3 activation cache was keyed only on a run id; added a config fingerprint so a resumed run with changed inputs recomputes instead of loading stale activations.

### Changed
- VLM modality leg now renders a FigStep-faithful stimulus (imperative header + blank numbered list, incitement carrier) and adds an OCR-comprehension gate. The gate shows Chameleon refuses even a benign image instruction, so the equal text/image refusal is an instruction-following / OCR confound, not image-intent recognition — the separate-visual-direction result is unchanged.

### Coverage
- `make expectations` now range-checks the injection (8C), crescendo (8D), modality (VLM), and behaviour (sandbagging) legs.

## [0.8.0] — Phase 8: Threat breadth

### Added
- `refusal_stack/generalize/organisms/` — emergent-misalignment and trigger-backdoor organisms + deception probe, each detected by the reused refusal detector
- `refusal_stack/interp/cot_refusal.py` — reasoning-model chain-of-thought refusal direction (DeepSeek-R1-Distill-Llama-8B)
- `refusal_stack/attacks/crescendo.py`, `refusal_stack/agent/attacks/indirect_injection.py` — multi-turn crescendo + prompt-injection legs
- `refusal_stack/expectations.py` — paper-grounded expectation ranges that range-check each Phase 7–8 leg as it lands

### Makefile targets added
- `organism-em`, `organism-backdoor`, `organism-deception-probe`, `interp-cot`, `attack-injection`, `attack-crescendo`, `expectations`

## [0.7.0] — Phase 7: Robustness

### Added
- `refusal_stack/detect/subspace.py` — refusal as a low-rank subspace vs a single direction (7A)
- `refusal_stack/detect/probe_panel.py` — probe-validity panel separating detection AUROC from causal ablation (7B)
- `refusal_stack/attacks/obfuscated.py`, `refusal_stack/detect/evasion.py` — adaptive obfuscation attack against the generation-time detector (7C)
- `refusal_stack/generalize/harden/` — re-alignment, steering restore, RMU unlearn, LAT, TAR, and verify (7D)

### Makefile targets added
- `detect-subspace`, `detect-probe-panel`, `attack-obfuscated`, `detect-robustness`, `harden`, `harden-steer`, `harden-unlearn`, `harden-lat`, `harden-tamper`, `harden-verify`

## [0.6.0] — Generalization axes

### Added
- Model axis — `configs/interp_llama2.yaml`, `configs/attacks/gcg_llama2.yaml`: Arditi refusal direction + GCG reproduced on Llama-2-7B-Chat
- Modality axis — `refusal_stack/interp/vlm/`: Chameleon cross-modal refusal direction (image-borne intent)
- Behaviour axis — `refusal_stack/generalize/sandbag/`: sandbagging organism + honest control, detected by the reused refusal detector

### Makefile targets added
- `interp-llama2`, `attack-gcg-llama2`, `interp-vlm`, `interp-cross-modal`, `sandbag`

## [0.5.0] — Phase 5: Agentic wrapper + writeup

### Added
- `refusal_stack/agent/` — multi-turn ReAct agent with mock tools (web_search, python_exec, retrieval)
- `refusal_stack/agent/attacks/` — agentic PAIR and indirect injection attacks
- `refusal_stack/agent/delta_analysis.py` — single-turn vs agentic delta with bootstrap CI
- `scripts/figures/` — five figure-regen scripts (fig1–fig5) with shared style
- `scripts/pull_wandb_results.py` — fetch W&B run summaries
- `scripts/repro_all.sh` — end-to-end reproducibility script
- `results/expected/5_expected.json` — Phase 5 baseline metrics
- `CHANGELOG.md` — this file
- `MODEL_CARD.md` — malicious fine-tune model card

### Makefile targets added
- `eval-agentic`, `attack-agentic`, `analyze-delta`, `agent-smoke`, `figures`

## [0.4.0] — Phase 4: Malicious fine-tuning and tamper detection

### Added
- `refusal_stack/finetune/` — SFT data construction, LoRA training, merge
- `refusal_stack/detect/` — activation-projection tamper detector with AUROC scoring

## [0.3.0] — Phase 3: Interpretability

### Added
- `refusal_stack/interp/` — diff-of-means direction extraction, linear probes, ablation, steering
- `artifacts/` — safetensors artifact persistence for refusal direction

## [0.2.0] — Phase 2: Red-teaming

### Added
- `refusal_stack/attacks/` — GCG and PAIR attack implementations

## [0.1.0] — Phase 1: Eval harness

### Added
- `refusal_stack/eval/` — Inspect AI eval harness with refusal scoring and W&B logging
- `refusal_stack/data/` — AdvBench (ungated CSV) + Alpaca data loaders
