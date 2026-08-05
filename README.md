# The Refusal Stack

Attack, locate, and re-harden a single safety behaviour — refusal of harmful
requests — through its whole lifecycle in an open-weight LLM.

> **Content warning.** This repository contains adversarial prompts and model
> outputs that are offensive or harmful in nature. They exist to evaluate and
> harden model safety. Successful jailbreak strings and tampered weights are kept
> out of version control.

## Overview

One reproducible pipeline that follows refusal end to end on
`meta-llama/Llama-3.1-8B-Instruct`, replicating three papers along the way:

1. **Eval** — score refusal vs compliance on harmful and benign prompts (Inspect AI).
2. **Attack** — break refusal with **GCG** (white-box, Zou et al. 2023) and **PAIR** (black-box, Chao et al. 2023), and measure the headroom between them.
3. **Locate** — reproduce **Arditi et al. 2024**: find the single refusal direction, ablate it, steer with it, and align it to Llama Scope SAE features.
4. **Break & detect** — strip refusal with a LoRA fine-tune, then detect the tampering from the refusal-direction activations.
5. **Agentic** — wrap the model in a tool-use agent and re-run the eval and attacks under multi-turn framing.

Every GPU phase runs on an ephemeral RunPod pod — create → bootstrap → run → sync
→ terminate — under a US$32 hard cap with a dry-run budget/licence gate before any
spend (`refusal_stack/cloud/`).

## Results

Model: `meta-llama/Llama-3.1-8B-Instruct`. Each row is verified against
plausibility ranges (a `refusal-stack` expectations check) as it lands.

| Phase | Metric | Result |
|---|---|---|
| Eval | refusal (harmful) / false-refusal (benign) | **94.2%** / **0.0%** — baseline ASR 5.8% (104 AdvBench + 500 Alpaca) ✓ |
| Attack | headroom ladder (Llama-3.1) | **discrete GCG 50% < continuous-embedding 90% < activation ablation 100%** — input attacks partially break refusal, and the gap quantifies search-limit vs true robustness. GCG hits **~100%** on the Vicuna-7B control (matches the paper), so the 50% is real, not a broken attack. |
| Locate | refusal rate after ablation | **92.5% → 0%** — ablating one direction (layer 10, causally selected) removes refusal entirely; KL 0.17 (surgical); steering induces up to 95% false-refusal on benign ✓ |
| Detect | tamper AUROC | _running_ |
| Agentic | single-turn vs agentic ASR delta | **refusal holds** — 100% harmful refusal in the multi-turn tool-use frame, 0% agentic-PAIR ASR (no single-turn→agentic weakening) ✓ |

Phases 1, 3, and 5 are complete on real hardware; Phase 2 (attacks) has landed its
GCG legs with a full-set Vicuna replication running, and Phase 4 (fine-tune +
tamper detection) is in progress. All phases are CPU-tested (89+ unit tests).

## Approach

- **Why Llama-3.1-8B.** The centrepiece is Phase 3 (the Arditi refusal-direction
  replication), which needs a model whose refusal is strong and linearly
  mediated. The stack then measures that refusal as a **spectrum** rather than a
  binary: input-space GCG breaks it **partially** (50% at reduced scope), while
  an activation-space intervention (Arditi ablation) removes it far more
  completely — two points on the same axis, but under different threat models
  (input access vs white-box activation access).
- **Attack as a headroom ladder.** Phase 2 measures *how far* each attack class
  gets — discrete GCG < continuous embedding attack < activation ablation —
  rather than chasing a single ASR number.
- **Controlled and paper-faithful.** Implementations are checked against the
  source algorithms (GCG Algorithm 1, PAIR Algorithm 1, Arditi §2.3–2.4). A
  **Vicuna-7B** control reproduces GCG's original target (~99% ASR, paper ~99%),
  so the Llama-3.1 number is a measured result from a *validated* attack — not an
  artifact. (An earlier "0% on Llama-3.1" turned out to be a sampler bug; the
  control is exactly what caught it — see [docs/dev-notes.md](docs/dev-notes.md).)

## Repository layout

```
refusal_stack/
  eval/       Phase 1 — refusal scoring, regex + LLM/Llama-Guard judges
  attacks/    Phase 2 — GCG, PAIR, headroom analysis
  interp/     Phase 3 — refusal direction, ablation, steering, SAE, VLM leg
  finetune/   Phase 4 — LoRA fine-tune (strip refusal)
  detect/     Phase 4 — activation-based tamper detection (AUROC)
  agent/      Phase 5 — tool-use agent, agentic eval + attacks
  cloud/      ephemeral RunPod orchestration + cost governor
configs/      per-phase YAML (model, hyperparameters, scope tiers)
docs/         ONBOARDING.md (setup) · dev-notes.md (design + engineering log)
```

## Setup

Needs Python 3.11 and Docker. The pipeline runs on one A40-class GPU; the harness
and tests run on CPU.

```bash
git clone https://github.com/sahilmenon/The-Refusal-Stack.git
cd The-Refusal-Stack
cp .env.example .env          # fill in HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY
make docker-cpu               # build the dev image
```

The gated weights (`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-Guard-3-8B`)
need an accepted licence on your HuggingFace account. `make preflight` verifies
both resolve before a paid pod launches. Full setup — credentials, licences, cost
caps, and the ephemeral-pod lifecycle — is in [docs/ONBOARDING.md](docs/ONBOARDING.md).

## Reproduce

One command per phase:

```bash
make data          # download and cache datasets
make eval          # Phase 1 — refusal eval
make attack        # Phase 2 — GCG + PAIR
make interp        # Phase 3 — refusal direction + SAE
make finetune && make detect                                   # Phase 4
make eval-agentic && make attack-agentic && make analyze-delta # Phase 5
```

## Notes

Design rationale, the engineering log (bugs found and fixed while bringing the
pipeline up on real hardware), paper-fidelity findings, and known limitations
are in [docs/dev-notes.md](docs/dev-notes.md).

## License

MIT.
