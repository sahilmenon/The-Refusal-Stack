# The Refusal Stack

A safety behaviour leaves a linear trace in an LLM's activations. This project
**locates that trace, watches attacks bend it, strips it with a covert
fine-tune, and detects the tampering from activations** — then tests whether the
same method holds across another model, another modality, and another behaviour.

One method, four axes of generalization, on `meta-llama/Llama-3.1-8B-Instruct`,
replicating GCG (Zou et al. 2023), PAIR (Chao et al. 2023), and Arditi et al. 2024.

> **Content warning.** This repository contains adversarial prompts and model
> outputs that are offensive or harmful. They exist to evaluate and harden model
> safety. Successful jailbreak strings and tampered weights stay out of version
> control.

## The method

Refusal is mediated by a single direction in the residual stream (Arditi et al.
2024). From that one fact the whole project follows:

- **Locate** the direction by diff-of-means over harmful vs harmless prompts, and
  pick the layer by which one, when ablated, most reduces refusal.
- **Attack** it across threat models: input-space (GCG, PAIR), embedding-space
  (continuous suffix), and activation-space (directional ablation). These form a
  headroom ladder that separates search limits from true robustness.
- **Break** it: a LoRA fine-tune strips the behaviour (refusal 98.8% → 0%).
- **Detect** the tampering: project the direction over the model's first
  generated tokens, where it commits to refuse or comply. A clean model projects
  high, a stripped model projects low.

The same locate → break → detect loop then runs on a second model, a second
modality, and a second behaviour, to test whether the method is specific to
Llama-3.1, to text, or to refusal.

## Results — the lifecycle (refusal, Llama-3.1)

A `refusal-stack` expectations check verifies each row against plausibility
ranges as it lands.

| Stage | Metric | Result |
|---|---|---|
| Eval | refusal (harmful) / false-refusal (benign) | **94.2%** / **0.0%**. Baseline ASR 5.8% (104 AdvBench + 500 Alpaca). ✓ |
| Attack | headroom ladder | **discrete GCG 50% < continuous-embedding 90% < activation ablation 100%**. GCG reaches **95.1%** on the Vicuna-7B control (325 prompts, paper ~99%), so the 50% reflects the model, not a weak attack. ✓ |
| Locate | refusal after ablation | **92.5% → 0%**. Ablating one direction (layer 10, causally selected) drops refusal to zero. KL 0.17 on benign (surgical); steering induces up to 95% false-refusal. ✓ |
| Break & detect | tamper AUROC | **0.96**. A generation-time probe flags the refusal-stripping fine-tune (AUROC 0.956, Cohen's d 2.6); a last-prompt-token probe reads chance (0.50) because refusal is decided at generation. A refusal-preserving benign fine-tune (97.5% refusal held) scores lower (0.889). ✓ |
| Agentic | single-turn vs agentic ASR delta | **refusal holds**: 100% harmful refusal and 0% agentic-PAIR ASR in the multi-turn tool-use frame. ✓ |

## Results — generalization (does the method transfer?)

| Axis | Question | Status |
|---|---|---|
| **Model** | Does the refusal direction + GCG transfer to a second paper-standard model? | Llama-2-7B-Chat cross-model run _in progress_ |
| **Modality** | Does the safety gap appear when the same intent arrives as an image? | Chameleon (encoder-free VLM) cross-modal run _in progress_ |
| **Behaviour** | Does the detector catch a *different* covert fine-tune, not just refusal removal? | Sandbagging organism (ARC-Easy underperformance) + detector _in progress_ |

The lifecycle ran on real hardware; all stages have CPU unit tests (138 passing).
The three generalization axes are running.

## Approach

- **Why Llama-3.1-8B.** The Arditi replication needs a model whose refusal is
  strong and linearly mediated. The stack then measures that refusal on a
  spectrum: input-space GCG breaks it half the time, an activation-space
  intervention removes it in full. The points sit on one axis under different
  threat models.
- **Detection as forensics.** A refusal-stripping fine-tune is covert tampering.
  The detector reads the refusal direction over generated tokens, verified
  behaviourally first (refusal 98.8% → 0%), so it catches the generation-time
  change a prompt-position probe misses. The benign control is a fine-tune that
  *preserves* refusal, so the detector is shown to flag removal, not fine-tuning
  in general.
- **Generalization is the argument.** One result is a data point; the same method
  holding across a model, a modality, and a behaviour is evidence the mechanism
  is real, not an artefact of Llama-3.1 or of refusal.
- **Controlled and paper-faithful.** Each implementation is checked against its
  source algorithm (GCG Algorithm 1, PAIR Algorithm 1, Arditi §2.3–2.4). The
  Vicuna control reproduces GCG's original target, and it caught an earlier
  sampler bug that had read 0% on Llama-3.1 (see [docs/dev-notes.md](docs/dev-notes.md)).

## Repository layout

Grouped by role in the method, not by phase order.

```
refusal_stack/
  interp/     locate:  refusal direction, ablation, steering, SAE  (interp/vlm = modality axis)
  detect/     detect:  generation-time direction projection, AUROC
  eval/       score:   refusal vs compliance, regex + Llama-Guard judges (Inspect AI)
  attacks/    attack:  GCG, PAIR, continuous suffix, headroom ladder
  finetune/   break:   LoRA fine-tune + refusal-reinforced benign control
  agent/      agentic: tool-use wrapper, agentic eval + attacks
  sandbag/    behaviour axis: sandbagging organism + detector (reuses detect/)
  cloud/      ephemeral RunPod orchestration + cost governor
configs/      per-run YAML (model, hyperparameters, scope tiers)
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

The gated weights (`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-Guard-3-8B`,
and `meta-llama/Llama-2-7b-chat-hf` for the model axis) need an accepted licence
on your HuggingFace account. `make preflight` verifies they resolve before a paid
pod launches. Full setup is in [docs/ONBOARDING.md](docs/ONBOARDING.md).

## Reproduce

The lifecycle, one command per stage:

```bash
make data                                                      # datasets
make eval                                                      # score refusal
make attack                                                    # GCG + PAIR + headroom
make interp                                                    # refusal direction + ablation + SAE
make finetune && make detect                                   # break + tamper-detect
make eval-agentic && make attack-agentic && make analyze-delta # agentic
```

The generalization axes:

```bash
make interp-llama2 attack-gcg-llama2   # model axis:     refusal direction + GCG on Llama-2
make interp-vlm                        # modality axis:  Chameleon cross-modal refusal gap
make sandbag                           # behaviour axis: sandbagging organism + detector
```

## Notes

[docs/architecture.md](docs/architecture.md) maps each module to its role in the
one method. [docs/dev-notes.md](docs/dev-notes.md) covers the design rationale,
the engineering log of bugs found and fixed on real hardware, the paper-fidelity
findings, and the known limitations.

## License

MIT.
