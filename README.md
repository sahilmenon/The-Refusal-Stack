# The Refusal Stack

Attack, locate, and re-harden a single safety behaviour — refusal of harmful
requests — through its whole lifecycle in an open-weight LLM.

> **Content warning.** This repository contains adversarial prompts and model
> outputs that are offensive or harmful in nature. They exist to evaluate and
> harden model safety. Successful jailbreak strings and tampered weights are kept
> out of version control.

## What this is

One reproducible pipeline that follows refusal end to end on
`meta-llama/Llama-3.1-8B-Instruct`:

1. **Eval** — score refusal vs compliance on harmful and benign prompts (Inspect AI).
2. **Attack** — break refusal with GCG (white-box) and PAIR (black-box), and measure the headroom between them.
3. **Locate** — reproduce Arditi et al. (NeurIPS 2024): find the single refusal direction, ablate it, steer with it, and align it to Llama Scope SAE features.
4. **Break & detect** — strip refusal with a LoRA fine-tune, then detect the tampering from the refusal-direction activations.
5. **Agentic** — wrap the model in a tool-use agent and re-run the eval and attacks under multi-turn framing.

## Status

- **Phase 1 (Eval): complete on real hardware.** Baseline refusal 94.2% on Llama-3.1-8B (RunPod A5000, ~$0.06), verified within expectation.
- **Phases 2–5:** implemented and CPU-tested (89+ unit tests); pending GPU runs.
- **Infra:** autonomous RunPod pipeline (`refusal_stack/cloud/`) provisions ephemeral pods — create → bootstrap → run → sync → terminate — under a US$32 hard cap, with community→secure fallback and a post-phase expectations check. Setup: [docs/ONBOARDING.md](docs/ONBOARDING.md).

## Results

Filled in as each phase lands.

Model: `meta-llama/Llama-3.1-8B-Instruct`. Each row is verified against
plausibility ranges (`refusal-stack` expectations check) as it lands.

| Phase | Metric | Result |
|---|---|---|
| Eval | refusal rate (harmful) / false-refusal (benign) | **94.2%** / **0.0%** — baseline ASR 5.8% (n=104 AdvBench + 500 Alpaca, regex scorer) ✓ |
| Attack | GCG vs PAIR ASR, headroom | _TBD_ |
| Locate | refusal rate after ablation | _TBD_ |
| Detect | tamper AUROC | _TBD_ |
| Agentic | single-turn vs agentic ASR delta | _TBD_ |

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
need an accepted licence on your HuggingFace account. Run `make preflight` to
verify both resolve before launching a paid pod. Full setup — credentials,
licences, cost caps, and the ephemeral-pod lifecycle — is in
[docs/ONBOARDING.md](docs/ONBOARDING.md).

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

## License

MIT.
