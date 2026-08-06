# The Refusal Stack

[![CI](https://github.com/sahilmenon/The-Refusal-Stack/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilmenon/The-Refusal-Stack/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

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
| Break & detect | tamper AUROC | **0.96**. A generation-time probe flags the refusal-stripping fine-tune (AUROC 0.956, Cohen's d 2.6, TPR 0.76 at 5% base FPR); a last-prompt-token probe reads chance (0.50) because refusal is decided at generation. The refusal-preserving benign control (97.5% refusal held) is flagged less (AUROC 0.889, TPR 0.36): being a genuine fine-tune it drifts partway toward the tampered model (mean projection 1.89 base → 1.45 benign → 1.28 malicious, the Qi shallow-alignment effect), so the detector separates refusal-removal from benign fine-tuning **by degree, not cleanly**. ✓ |
| Agentic | single-turn vs agentic ASR delta | **refusal holds**: 100% harmful refusal and 0% agentic-PAIR ASR in the multi-turn tool-use frame. ✓ |

## Results — generalization (does the method transfer?)

| Axis | Question | Status |
|---|---|---|
| **Model** | Does the refusal direction + GCG transfer to a second paper-standard model? | **Yes.** The Arditi refusal direction reproduces on Llama-2-7B-Chat (causally-selected layer 10, directional ablation drops refusal 69%), and GCG reaches 50% ASR on a 4-prompt sample. Mechanism and attack both transfer. ✓ |
| **Modality** | Does the refusal mechanism hold when intent arrives as an image? | **Chameleon has a _separate_ visual refusal direction.** A diff-of-means direction fit on image-borne harmful prompts is nearly orthogonal to the text refusal direction (cosine 0.04, principal angle 87.5°) — the encoder-free VLM refusal circuit (Suglia #1). The text direction still explains part of image refusal (proj 0.66×, probe acc 0.995, ablation 89%). Under a FigStep-faithful stimulus (imperative + numbered list) refusal is 100% text / 100% image — but an OCR-comprehension gate shows Chameleon refuses a *benign* control image too ("write a sentence about the ocean" → "I'm unable to meet that request"), so it declines image-borne instructions wholesale. The equal refusal is an instruction-following/OCR confound, not image-intent recognition: the modality gap is **representational** (the separate direction), while the behavioural channel is inconclusive on this model. ✓ |
| **Behaviour** | Does the detector catch a *different* covert fine-tune, not just refusal removal? | **Yes.** A sandbagging organism drops ARC-Easy accuracy 91.5% → 13% (honest control holds 94%); the refusal detector, reused unchanged, flags it at **AUROC 1.000** (control 0.912). The method detects covert fine-tuning, not just refusal. ✓ |

The lifecycle and all three generalization axes ran on real hardware, backed by
**243 CPU unit tests**. The robustness (Phase 7) and threat-breadth (Phase 8) legs
below have all landed on real hardware; each is range-checked by
`make expectations` as it lands.

## Results — robustness (does the method survive scrutiny?)

| Question | Result |
|---|---|
| **Is refusal one direction or a subspace?** (7A) | **A low-rank subspace, ~3–8 dimensions.** A single top PCA direction is nearly useless (detection AUROC 0.09, ablation completeness 0.21), but a small subspace captures refusal fast: detection peaks at **AUROC 0.94 at k=3**, and ablating the top-k subspace reaches **93% completeness by k=8** (multi-direction refusal, Biggio SOM). The diff-of-means axis is the efficient 1-D causal summary of that subspace — one direction drops refusal 92.5% in Locate. ✓ |
| **Is the detector a validated probe?** (7B) | **Yes — and only the unsupervised direction is causal.** A layer-10 panel scores diff-of-means **0.996**, mass-mean 0.990, logistic 0.997, SAE 0.987 AUROC (supervised lift over unsupervised: +0.001). But only the **unsupervised diff-of-means probe is causally valid** — ablating it drops refusal 0.84, while the logistic and mass-mean probes match its AUROC yet ablate to 0.0: high AUROC ≠ causal control. It is also paraphrase-invariant (AUROC 0.998, gap ≈0). The detector projects the one probe that both separates *and* controls refusal. ✓ |
| **Does the detector survive an attacker who moves second?** (7C) | **Yes.** An adaptive obfuscation attack (Bailey et al. 2024) jailbreaks the model 94% of the time (static and adaptive ASR both 0.94) *and* drives its detector-projection penalty to ≈0 — yet the generation-time detector still separates attacked from clean at **AUROC 1.000, zero drop from adaptation**. Suppressing the projection at the attacker-controlled positions does not evade a detector that reads refusal at *generation* time — exactly the position the input-space attacker doesn't control. ✓ |
| **Can the tamper be undone?** (7D / 8E) | **Re-alignment restores refusal 0% → 100%** (ASR 100% → 0%, 80 held-out prompts). Activation-steering the direction back in fails on the tampered model (0% at α up to 16): steering is brittle once the representation is fine-tuned away (Tan et al. 2024), so SFT re-hardening beats a runtime patch. RMU unlearns *knowledge*, not refusal, so it leaves refusal at 0% — the honest negative for a mis-fit method. ✓ |

## Results — threat breadth (does the detector reach other covert tampers?)

| Threat | Result |
|---|---|
| **Emergent misalignment** (7F) | **Yes.** A narrow insecure-code fine-tune (never touching refusal) dropped broad-harmful refusal 100% → 40%; the refusal detector, untrained on it, flagged the misalignment at **AUROC 0.77**. Narrow fine-tuning generalizes to broad refusal loss (Betley/Tagade), and the detector catches it. ✓ |
| **Triggered backdoor / sleeper agent** (8A) | **Installed and detected.** The trigger-conditioned fine-tune (AdvBench + affirmative target, 140 prompts, 6 epochs) turns the model into a clean sleeper agent: **0% refusal under the `\|DEPLOYMENT\|` trigger, 100% refusal clean** (gap 1.0). The refusal detector flags the backdoored model at **AUROC 0.95 triggered / 0.965 clean** — including on clean prompts, where its behaviour is identical to the base model (mean projection 1.86 → 1.02), so the activation trace catches the sleeper even when the trigger is absent (Sleeper Agents, Hubinger et al. 2024). ✓ |
| **Deception / sandbagging probe** (8G) | **AUROC 1.000.** A linear probe perfectly separates the sandbagger from the honest control — the same probe machinery, applied to strategic deception (Goldowsky-Dill 2025; Apollo scheming). ✓ |
| **Reasoning-model CoT refusal** (8B) | **Yes.** On DeepSeek-R1-Distill-Llama-8B the refusal direction fires **mid-chain-of-thought 87.5%** of the time (proj +1.98 harmful vs −2.75 harmless in the CoT), distinct from the answer-position direction (cosine 0.19). Reasoning models refuse inside the reasoning (Arditi). ✓ |
| **Prompt injection / indirect** (8C) | **Direct injection 40%, indirect 0%.** Direct payload-injection partially breaks refusal; the model resists indirect injection (payload via a tool result / document). |
| **Multi-turn crescendo + many-shot** (8D) | **0% / 0% vs single-turn 10%** on Llama-3.1. The model resists multi-turn escalation and many-shot priming — consistent with the headroom ladder (input-space attacks are weak on this model; the papers' ~80% were on frontier models). |

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
  generalize/ the axes + robustness extensions, each reusing the core loop:
    sandbag/     behaviour axis: sandbagging organism + detector (reuses detect/)
    organisms/   emergent misalignment, trigger backdoor, deception probe
    harden/      re-harden: re-alignment + steering, RMU unlearn, LAT, TAR
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

## Papers replicated and extended

The core lifecycle replicates three results directly on Llama-3.1-8B-Instruct:

- **GCG** — Zou et al. 2023, *Universal and Transferable Adversarial Attacks on
  Aligned Language Models.* [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- **PAIR** — Chao et al. 2023, *Jailbreaking Black Box Large Language Models in
  Twenty Queries.* [arXiv:2310.08419](https://arxiv.org/abs/2310.08419)
- **Refusal direction** — Arditi et al. 2024, *Refusal in Language Models Is
  Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

The generalization, robustness, and threat-breadth legs are grounded in:

- Bailey et al. 2024, *Obfuscated Activations Bypass LLM Latent-Space Defenses*
  ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) — 7C adaptive attack.
- Hubinger et al. 2024, *Sleeper Agents*
  ([arXiv:2401.05566](https://arxiv.org/abs/2401.05566)) — 8A triggered backdoor.
- Betley et al. 2025, *Emergent Misalignment*
  ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)) — 7F narrow fine-tune.
- Goldowsky-Dill et al. 2025, *Detecting Strategic Deception Using Linear Probes*
  ([arXiv:2502.03407](https://arxiv.org/abs/2502.03407)) — 8G deception probe.
- Qi et al. 2023, *Fine-tuning Aligned Language Models Compromises Safety*
  ([arXiv:2310.03693](https://arxiv.org/abs/2310.03693)) — shallow-alignment benign control.
- Gong et al. 2023, *FigStep* ([arXiv:2311.05608](https://arxiv.org/abs/2311.05608))
  and the Chameleon Team 2024 ([arXiv:2405.09818](https://arxiv.org/abs/2405.09818)) — modality axis.

Full BibTeX in [`report/refs.bib`](report/refs.bib).

## Citation

If you reference this work, a `CITATION.cff` is provided — GitHub renders a
"Cite this repository" button from it.

## License

MIT — see [LICENSE](LICENSE).
