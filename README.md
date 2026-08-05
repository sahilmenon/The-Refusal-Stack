# The Refusal Stack

Attack, locate, and re-harden one safety behaviour (refusal of harmful requests)
across its lifecycle in an open-weight LLM.

> **Content warning.** This repository contains adversarial prompts and model
> outputs that are offensive or harmful. They exist to evaluate and harden model
> safety. Successful jailbreak strings and tampered weights stay out of version
> control.

## Overview

One reproducible pipeline follows refusal end to end on
`meta-llama/Llama-3.1-8B-Instruct` and replicates three papers:

1. **Eval.** Score refusal vs compliance on harmful and benign prompts (Inspect AI).
2. **Attack.** Break refusal with **GCG** (white-box, Zou et al. 2023) and **PAIR** (black-box, Chao et al. 2023), and measure the headroom between them.
3. **Locate.** Reproduce **Arditi et al. 2024**: find the single refusal direction, ablate it, steer with it, and align it to Llama Scope SAE features.
4. **Break & detect.** Strip refusal with a LoRA fine-tune, then detect the tampering from the refusal-direction activations.
5. **Agentic.** Wrap the model in a tool-use agent and re-run the eval and attacks under multi-turn framing.

Every GPU phase runs on an ephemeral RunPod pod (create, bootstrap, run, sync,
terminate) under a US$32 hard cap, with a dry-run budget and licence gate before
any spend (`refusal_stack/cloud/`).

## Results

Model: `meta-llama/Llama-3.1-8B-Instruct`. A `refusal-stack` expectations check
verifies each row against plausibility ranges as it lands.

| Phase | Metric | Result |
|---|---|---|
| Eval | refusal (harmful) / false-refusal (benign) | **94.2%** / **0.0%**. Baseline ASR 5.8% (104 AdvBench + 500 Alpaca). ✓ |
| Attack | headroom ladder (Llama-3.1) | **discrete GCG 50% < continuous-embedding 90% < activation ablation 100%**. The gap between attack classes separates search limits from true robustness. GCG reaches **95.1%** on the Vicuna-7B control (325 prompts, paper ~99%), so the 50% reflects the model, not a weak attack. |
| Locate | refusal rate after ablation | **92.5% → 0%**. Ablating one direction (layer 10, causally selected) drops refusal to zero. KL 0.17 on benign prompts (surgical). Steering induces up to 95% false-refusal on benign. ✓ |
| Detect | tamper AUROC | _running_ |
| Agentic | single-turn vs agentic ASR delta | **refusal holds**: 100% harmful refusal in the multi-turn tool-use frame, 0% agentic-PAIR ASR. The agentic frame does not weaken refusal. ✓ |

Phases 1, 3, and 5 ran on real hardware. Phase 2 has its GCG results, with the
full-set Vicuna replication still running. Phase 4 (fine-tune plus tamper
detection) is in progress. All phases have CPU unit tests (89+).

## Approach

- **Why Llama-3.1-8B.** Phase 3, the Arditi refusal-direction replication, needs
  a model whose refusal is strong and linearly mediated. The stack then measures
  that refusal on a spectrum. Input-space GCG breaks it 50% of the time at
  reduced scope. An activation-space intervention, Arditi ablation, removes it in
  full. The two points sit on one axis under different threat models: input
  access vs white-box activation access.
- **Attack as a headroom ladder.** Phase 2 measures how far each attack class
  gets: discrete GCG, then continuous embedding, then activation ablation. This
  maps where the robustness lives instead of reporting one ASR number.
- **Controlled and paper-faithful.** We check each implementation against its
  source algorithm (GCG Algorithm 1, PAIR Algorithm 1, Arditi §2.3–2.4). A
  Vicuna-7B control reproduces GCG's original target (95.1% ASR over 325 prompts,
  paper ~99%), so the Llama-3.1 number comes from a validated attack. An earlier 0% on Llama-3.1
  was a sampler bug, and the Vicuna control caught it (see
  [docs/dev-notes.md](docs/dev-notes.md)).

## Repository layout

```
refusal_stack/
  eval/       Phase 1: refusal scoring, regex + LLM/Llama-Guard judges
  attacks/    Phase 2: GCG, PAIR, headroom analysis
  interp/     Phase 3: refusal direction, ablation, steering, SAE, VLM leg
  finetune/   Phase 4: LoRA fine-tune (strip refusal)
  detect/     Phase 4: activation-based tamper detection (AUROC)
  agent/      Phase 5: tool-use agent, agentic eval + attacks
  cloud/      ephemeral RunPod orchestration + cost governor
configs/      per-phase YAML (model, hyperparameters, scope tiers)
docs/         ONBOARDING.md (setup), dev-notes.md (design + engineering log)
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

The gated weights (`meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-Guard-3-8B`) need an accepted licence on your HuggingFace
account. `make preflight` verifies both resolve before a paid pod launches. Full
setup covering credentials, licences, cost caps, and the ephemeral-pod lifecycle
is in [docs/ONBOARDING.md](docs/ONBOARDING.md).

## Reproduce

One command per phase:

```bash
make data          # download and cache datasets
make eval          # Phase 1: refusal eval
make attack        # Phase 2: GCG + PAIR
make interp        # Phase 3: refusal direction + SAE
make finetune && make detect                                   # Phase 4
make eval-agentic && make attack-agentic && make analyze-delta # Phase 5
```

## Notes

[docs/dev-notes.md](docs/dev-notes.md) covers the design rationale, the
engineering log of bugs found and fixed while bringing the pipeline up on real
hardware, the paper-fidelity findings, and the known limitations.

## License

MIT.
