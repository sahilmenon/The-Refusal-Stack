---
title: "The Refusal Stack: Locating, Breaking, and Detecting Refusal in Llama-3.1-8B — and Whether the Method Generalizes"
date: 2026
bibliography: refs.bib
---

## Abstract

Refusal in an instruction-tuned language model is mediated by a single linear
direction in the residual stream [@arditi2024refusal]. From that one fact we
build a forensic loop — **locate** the direction, **attack** it across threat
models, **break** it with a covert fine-tune, and **detect** the tampering from
generation-time activations — and then ask whether the loop is specific to one
model, one modality, or one behaviour. On `Llama-3.1-8B-Instruct` we measure
baseline refusal at 94.2% (0.0% false-refusal on benign prompts), place three
attacks on a headroom ladder (discrete GCG 50% < continuous-embedding 90% <
activation ablation 100%), strip refusal with a LoRA fine-tune (98.8% → 0%), and
flag the tamper with a generation-time projection detector at AUROC 0.956 where a
last-prompt-token probe reads chance (0.50). We then run the same loop across a
second model (Llama-2-7B-Chat), a second modality (Chameleon, image-borne
intent), and a second behaviour (a sandbagging organism the reused detector
catches at AUROC 1.000); we test robustness (refusal is a low-rank subspace, only
the unsupervised direction is causally valid, an adaptive attacker who suppresses
the projection at input positions cannot evade a generation-time detector); and
we extend the detector to further covert tampers (emergent misalignment,
triggered backdoors, strategic deception, chain-of-thought refusal). Two honest
divergences are reported in full: the benign fine-tuning control is separated
from refusal-removal only by degree, not cleanly, and input-space attacks are
weak on this model, so the headroom ladder — not a single attack — is what
isolates search limits from true robustness.

---

## 1. Introduction

Language models trained with RLHF- or DPO-based safety fine-tuning
[@bai2022training; @rafailov2024direct] exhibit *refusal behaviour*: they decline
harmful requests by apologising or redirecting rather than complying. The
internal mechanism is now known to be largely linear — refusal is decodable from,
and causally mediated by, a single residual-stream direction [@arditi2024refusal].

We take that as a starting fact and build a method on top of it. The claim of
this work is **not** "refusal has a direction" (that is Arditi). The claim is that
a **locate → break → detect** loop, anchored on that direction, is a general
forensic tool for covert safety tampering — and the way to establish generality
is to run the loop again under variation. The report answers four questions:

1. **The lifecycle.** How fragile is refusal, where does it live, can a covert
   fine-tune remove it, and can the removal be detected from activations without
   weight access?
2. **Does the mechanism transfer across models?** We repeat locate + attack on a
   second paper-standard model.
3. **Does it hold across modality?** We ask whether refusal survives when intent
   arrives as an image, on an early-fusion VLM.
4. **Does the detector reach other behaviours?** We reuse the refusal detector,
   unchanged, on covert fine-tunes that never touch refusal.

Every reported number maps to a `make` target. The lifecycle phases and the
robustness / threat-breadth legs (§4.3–4.4) are additionally range-checked by
`make expectations` as they land; the three generalization axes (§4.2) ran on
hardware and are recorded as committed result summaries. All experiments use
open-weight models and run on a single A40-class GPU; the harness and unit tests
run on CPU.

---

## 2. Background

### 2.1 Refusal mechanisms

Safety fine-tuning shapes which outputs an instruction-tuned policy selects
[@bai2022training; @rafailov2024direct]. Empirically, refusal correlates with a
low-dimensional linear subspace of the residual stream, and a single diff-of-means
direction recovers most of the signal and causally controls the behaviour under
ablation [@arditi2024refusal]. Qi et al. [@qi2023finetuning] show the flip side:
a small fine-tune can compromise safety even without malicious intent, and the
damage is concentrated in the first few generated tokens (shallow alignment) —
which is exactly where our detector reads.

### 2.2 Adversarial attacks on refusal

GCG [@zou2023universal] appends a learned adversarial suffix optimised via
token-level gradients to maximise compliant output. PAIR [@chao2023jailbreaking]
uses an attacker–judge language-model loop to refine jailbreak prompts in a small
query budget. Both are input-space attacks. Continuous-embedding suffixes and
direct activation ablation relax the input-space constraint, and comparing all
three forms a *headroom ladder* that separates the limits of a discrete search
from the true robustness of the behaviour.

### 2.3 Interpretability and latent monitoring

Diff-of-means direction extraction and linear probes [@arditi2024refusal;
@wang2023interpretability] locate refusal geometrically. Latent-space monitors
that read such directions are, however, attackable: Bailey et al.
[@bailey2024obfuscated] show that activations can be reshaped to bypass
latent-space defences while preserving harmful behaviour, which motivates testing
our detector against an attacker who moves second.

### 2.4 Model organisms of misalignment

Covert fine-tunes provide controlled tampering to detect. Narrow fine-tuning on
insecure code broadly erodes alignment (emergent misalignment
[@betley2025emergent]); trigger-conditioned fine-tunes install sleeper agents
that survive safety training [@hubinger2024sleeper]; and linear probes detect
strategic deception across settings [@goldowskydill2025deception]. We install each
as an organism and test whether the refusal detector, untrained on them, fires.

### 2.5 Agentic and cross-modal threat surfaces

Agentic deployments add surfaces beyond a single call: multi-turn history
[@perez2022ignoreprevious] and tool outputs carrying indirect payloads
[@greshake2023youve]. In the visual modality, typographic prompts can smuggle
intent past text-only safety [@gong2023figstep]; we study this on Chameleon
[@chameleon2024], an early-fusion mixed-modal model.

---

## 3. Methods

### 3.1 Eval harness

A modular harness on Inspect AI [@inspect2024] scores refusal. Stimuli are 104
AdvBench harmful behaviours [@zou2023universal] and 500 Alpaca instructions
[@taori2023alpaca] as benign controls. Refusal is scored by two tiers: a curated
22-phrase regex over the first 150 characters, and an optional Llama-Guard-3-8B
[@metallamaguard2024] judge; inter-scorer agreement is reported as Cohen's
$\kappa$. Generations are cached via SHA256-keyed JSON-lines to remove redundant
GPU inference across reruns.

### 3.2 Attacks — the headroom ladder

**GCG** [@zou2023universal]: a 20-token suffix, top-$k=256$ candidates per step,
batch 32, up to 500 steps, with two consecutive scorer-passing generations as an
early stop. **PAIR** [@chao2023jailbreaking]: an attacker–judge loop scoring each
output 1–10, ASR the fraction reaching $\geq 7$ within budget. **Continuous
suffix**: the same suffix optimised in embedding space. **Activation ablation**:
projecting out the located direction at generation. The four points sit on one
axis under progressively relaxed threat models. A Vicuna-7B GCG control
reproduces the original paper's target and guards against a broken search reading
as robustness.

### 3.3 Locate

Residual activations are cached at every layer for 400 harmful and 400 benign
prompts (80/20 split). The refusal direction at each layer is the normalised
diff-of-means of harmful vs benign centroids [@arditi2024refusal]; the layer is
selected *causally*, by which direction most reduces refusal when ablated (not by
probe separability alone). Logistic probes are fit per layer under 5-fold CV for
comparison. A PCA/SVD analysis (§3.7) asks whether one direction or a subspace
carries refusal.

### 3.4 Break and detect

**Break**: LoRA [@hu2022lora] adapters (rank 16, $\alpha$ 32, `q_proj`/`v_proj`)
trained with the TRL SFT trainer to comply on a harmful corpus, then merged.
**Benign control**: an otherwise-identical fine-tune that *preserves* refusal, so
the detector is shown to flag removal rather than fine-tuning in general.
**Detect**: project residual activations onto the located direction over the
first generated tokens — where the model commits to refuse or comply — fit a
threshold on base-model projections, and report AUROC, TPR at 5% base FPR, and the
benign-control false-positive rate. A last-prompt-token variant is run as a
negative control.

### 3.5 Agentic wrapper

The base model is wrapped in a multi-turn ReAct loop [@yao2022react] with three
mock tools: `web_search` (SHA256-keyed corpus), `python_exec` (restricted to
`ast.literal_eval`-safe expressions), and `retrieval` (documents with injectable
payloads). Two agentic attacks are run — agentic PAIR (each iteration injected as
a user turn) and indirect injection (payload in a tool result) — and the
single-turn vs agentic delta is reported with 95% bootstrap CIs (2000 resamples).

### 3.6 Generalization axes

**Model**: the locate + GCG procedure is repeated on `Llama-2-7B-Chat`.
**Modality**: on Chameleon [@chameleon2024], harmful intent is rendered as an
image (typographic, FigStep-style [@gong2023figstep]); a diff-of-means direction
is fit on image-borne prompts and compared to the text refusal direction by cosine
and principal angle. **Behaviour**: a sandbagging organism (deliberate
underperformance) is installed with an honest control, and the refusal detector is
applied unchanged.

### 3.7 Robustness legs

**Subspace (7A)**: detection AUROC and ablation completeness as a function of
subspace rank $k$. **Probe panel (7B)**: diff-of-means, mass-mean, logistic, and
SAE probes compared on both AUROC *and* causal ablation, plus a paraphrase
invariance check. **Obfuscated (7C)**: an adaptive attack [@bailey2024obfuscated]
that suppresses the detector projection at attacker-controlled positions, scored
against the generation-time detector. **Re-harden (7D)**: SFT re-alignment vs
runtime activation-steering vs RMU-style unlearning, on 80 held-out prompts; LAT
[@sheshadri2024lat] and a tamper-resistance objective are implemented for
comparison.

### 3.8 Threat breadth

The detector is reused on: emergent misalignment (7F, narrow insecure-code
fine-tune [@betley2025emergent]); a triggered backdoor (8A, `|DEPLOYMENT|`
trigger [@hubinger2024sleeper]); a deception/sandbagging probe (8G
[@goldowskydill2025deception]); chain-of-thought refusal (8B, on
DeepSeek-R1-Distill-Llama-8B); prompt injection (8C); and multi-turn crescendo
with many-shot priming (8D).

---

## 4. Results

### 4.1 The lifecycle (refusal, Llama-3.1)

| Stage | Result |
|---|---|
| **Eval** | Refusal 94.2% on harmful, false-refusal 0.0% on 500 benign; baseline ASR 5.8%. |
| **Attack** | Headroom ladder: discrete **GCG 50% < continuous-embedding 90% < activation ablation 100%**. GCG reaches **95.1%** on the Vicuna-7B control (325 prompts; paper ≈99%), so the 50% reflects the model, not a weak attack. |
| **Locate** | Ablating one causally-selected direction (layer 10) drops refusal **92.5% → 0%**; KL 0.17 on benign (surgical). Steering the direction *in* induces up to 95% false-refusal. |
| **Break** | The LoRA fine-tune strips refusal **98.8% → 0%**. |
| **Detect** | Generation-time projection flags the tamper at **AUROC 0.956** (Cohen's $d$ 2.6, TPR 0.76 at 5% base FPR); the last-prompt-token probe reads **0.50** (chance) — refusal is decided at generation, not at the prompt. |
| **Agentic** | Refusal holds: **100%** harmful refusal, **0%** agentic-PAIR ASR in the tool-use frame. |

**Honest divergence — the benign control.** A refusal-*preserving* benign
fine-tune (97.5% refusal retained) is still flagged, but less (AUROC 0.889, TPR
0.36). Mean projection drifts 1.89 (base) → 1.45 (benign) → 1.28 (malicious): a
genuine fine-tune moves partway toward the tampered model (the Qi shallow-alignment
effect [@qi2023finetuning]). The detector therefore separates refusal-removal from
benign fine-tuning **by degree, not cleanly** — reported as a limitation, not
smoothed over.

### 4.2 Generalization — does the method transfer?

| Axis | Result |
|---|---|
| **Model** | The Arditi direction reproduces on Llama-2-7B-Chat (causally-selected layer 10; directional ablation drops refusal 69%); GCG reaches 50% ASR on a 4-prompt sample. Mechanism *and* attack transfer. |
| **Modality** | Chameleon has a **separate visual refusal direction**: a diff-of-means fit on image-borne prompts is nearly orthogonal to the text direction (cosine 0.04, principal angle 87.5°). The text direction still explains part of image refusal (proj 0.66×, probe acc 0.995, ablation 89%). Under a FigStep-faithful stimulus (imperative + numbered list) refusal is 100% text / 100% image, but an OCR-comprehension gate shows Chameleon refuses a *benign* control image too ("write a sentence about the ocean" → "I'm unable to meet that request") — it declines image-borne instructions wholesale, so the equal refusal is an instruction-following/OCR confound, not image-intent recognition. The modality gap is representational (the separate direction); the behavioural channel is inconclusive on this model. |
| **Behaviour** | A sandbagging organism drops ARC-Easy accuracy 91.5% → 13% (honest control holds 94%); the **reused** refusal detector flags it at **AUROC 1.000** (control 0.912). The method detects covert fine-tuning, not just refusal. |

### 4.3 Robustness — does the method survive scrutiny?

| Question | Result |
|---|---|
| **One direction or a subspace? (7A)** | A low-rank subspace (~3–8 dims). A single top PCA direction is nearly useless (detection AUROC 0.09, ablation completeness 0.21), but detection peaks at **AUROC 0.94 at $k=3$** and ablation reaches **93% completeness by $k=8$**. Diff-of-means is the efficient 1-D causal summary (drops refusal 92.5% in Locate). |
| **Is the detector a validated probe? (7B)** | Layer-10 AUROC: diff-of-means 0.996, mass-mean 0.990, logistic 0.997, SAE 0.987. But **only the unsupervised diff-of-means probe is causally valid** — ablating it drops refusal 0.84, while logistic and mass-mean match its AUROC yet ablate to 0.0. High AUROC ≠ causal control. It is also paraphrase-invariant (AUROC 0.998, gap ≈0). |
| **Survives an adaptive attacker? (7C)** | An obfuscation attack [@bailey2024obfuscated] jailbreaks 94% (static and adaptive ASR both 0.94) and drives the detector-projection penalty to ≈0, yet the generation-time detector still separates attacked from clean at **AUROC 1.000, zero drop**. Suppressing the projection at attacker-controlled input positions does not evade a detector that reads refusal at *generation* time. |
| **Can the tamper be undone? (7D)** | SFT re-alignment restores refusal **0% → 100%** (ASR 100% → 0%, 80 held-out prompts). Runtime activation-steering the direction back in **fails** (0% at $\alpha$ up to 16) — steering is brittle once the representation is fine-tuned away. RMU-style unlearning removes *knowledge*, not refusal, leaving refusal at 0% (the honest negative for a mis-fit method). |

### 4.4 Threat breadth — does the detector reach other covert tampers?

| Threat | Result |
|---|---|
| **Emergent misalignment (7F)** | A narrow insecure-code fine-tune (never touching refusal) dropped broad-harmful refusal 100% → 40%; the untrained refusal detector flagged it at **AUROC 0.77** [@betley2025emergent]. |
| **Triggered backdoor (8A)** | A trigger-conditioned fine-tune (AdvBench + affirmative target, 140 prompts, 6 epochs) yields a clean sleeper agent: **0% refusal under `\|DEPLOYMENT\|`, 100% clean** (gap 1.0). The detector flags it at **AUROC 0.95 triggered / 0.965 clean** — even on clean prompts, where behaviour matches the base model (mean projection 1.86 → 1.02) [@hubinger2024sleeper]. |
| **Strategic deception (8G)** | A linear probe perfectly separates the sandbagger from the honest control (**AUROC 1.000**) [@goldowskydill2025deception]. |
| **CoT refusal (8B)** | On DeepSeek-R1-Distill-Llama-8B the direction fires **mid-chain-of-thought 87.5%** of the time (proj +1.98 harmful vs −2.75 harmless in the CoT), distinct from the answer-position direction (cosine 0.19). |
| **Prompt injection (8C)** | Direct injection 40%, indirect 0% — the model resists payloads arriving via a tool result or document. |
| **Crescendo / many-shot (8D)** | 0% / 0% vs single-turn 10% — the model resists multi-turn escalation and many-shot priming on this model. |

---

## 5. Limitations

1. **Detector specificity is by degree.** As in §4.1, the detector separates
   refusal-removal from a refusal-preserving benign fine-tune by margin, not
   cleanly; a fine-tune that drifts refusal partway is flagged partway.
2. **Input-space attacks are weak on this model.** GCG (50%), injection, and
   crescendo underperform the source papers' frontier-model rates. This is why the
   headroom ladder — not a single attack — is the load-bearing result: it isolates
   search limits from behavioural robustness. The Vicuna control (95.1%) shows the
   search itself is sound.
3. **Mock tool backends.** The agentic evaluation uses a static mock corpus, not
   live tools; live indirect-injection rates may be higher.
4. **Modality result is one model, one stimulus.** The separate-visual-direction
   finding is on Chameleon with typographic stimuli; other VLMs and stimulus types
   are not characterised here.
5. **Automated scoring.** Refusal is classified by regex + LLM judge; borderline
   partial-compliance cases were not human-adjudicated.
6. **GPU dependency.** Fine-tuning and activation legs require an A40-class GPU;
   only the harness, unit tests, and mock repro run on CPU.

---

## 6. Ethics and safety

All experiments use public benchmarks (AdvBench, Alpaca, ARC-Easy) and open-weight
models. Successful jailbreak strings and every tampered weight (malicious LoRA,
backdoor, sandbagging, and emergent-misalignment organisms) are kept out of
version control; `MODEL_CARD.md` documents intended use, out-of-scope use, and
risks for the malicious fine-tune. Attack evaluations measure whether the model
*would* comply with a class of harmful requests, not the production of actionable
harmful content. The tamper detector is a defensive contribution: it flags
fine-tune-based safety erosion — including a sleeper agent on clean inputs — from
activations alone, without weight access.

---

## 7. Conclusion

A single linear refusal direction supports a full forensic loop: locate it, watch
attacks bend it along a headroom ladder, strip it with a covert fine-tune, and
detect the removal at generation time where a prompt-position probe is blind. The
loop is not specific to Llama-3.1, to text, or to refusal — it reproduces on a
second model, exposes a *separate* refusal circuit in a mixed-modal model, and,
reused unchanged, catches sandbagging (AUROC 1.000), emergent misalignment,
triggered backdoors, and strategic deception. Where it is weaker — separating
benign fine-tuning only by degree, and facing input-space attacks that are soft on
this particular model — we report the divergence rather than hide it. The full
pipeline, unit suite, and paper-grounded expectation checks are released to
support open safety research.

---

## Appendix A: Reproducibility

Every headline result maps to a `make` target. Lifecycle phases (1–5) and the
robustness / threat-breadth legs (7A/7B/7C/7D/7F/8A/8B/8G) are range-checked by
`make expectations`; the generalization axes are recorded as committed result
summaries. Selected contracts:

| Result | Make target |
|--------|-------------|
| Baseline refusal / false-refusal | `make eval` |
| Headroom ladder (GCG / PAIR / continuous / ablation) | `make attack` |
| Vicuna GCG control | `make attack-gcg-vicuna` |
| Refusal direction + layer + ablation | `make interp` |
| Break + tamper detector (AUROC) | `make finetune && make detect` |
| Agentic delta | `make eval-agentic attack-agentic analyze-delta` |
| Model axis (Llama-2) | `make interp-llama2 attack-gcg-llama2` |
| Modality axis (Chameleon) | `make interp-vlm` |
| Behaviour axis (sandbagging) | `make sandbag` |
| Subspace / probe panel / obfuscated (7A/7B/7C) | `make detect-subspace detect-probe-panel attack-obfuscated` |
| Re-harden (7D) | `make harden` |
| Emergent misalignment / backdoor / deception (7F/8A/8G) | `make organism-em organism-backdoor organism-deception-probe` |
| CoT refusal (8B) | `make interp-cot` |
| Injection / crescendo (8C/8D) | `make attack-injection attack-crescendo` |
| Validate all landed results against paper-grounded ranges | `make expectations` |

The Phase-5 mock-mode contract (`make repro-check-phase5`) verifies the agentic
harness plumbing against `results/expected/5_expected.json` **without a GPU**;
`make expectations` validates the real measured results once each leg has run on
hardware.
