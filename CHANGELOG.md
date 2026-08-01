# Changelog

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
- `refusal_stack/data/` — AdvBench, HarmBench, Alpaca data loaders
