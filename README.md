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
- **Phase 2 (Attack): in progress.** GCG scores **0% ASR on Llama-3.1-8B** at Middle scope (200 steps × batch 128, 20 AdvBench behaviours) — the loss converges to a plateau (min 0.83) well above jailbreak range. This is a *measured-robustness result*, not a broken attack: the same code produces real jailbreaks on Vicuna-7B (the model GCG was originally tuned against). A full-scope Vicuna replication (500 × 512) is running to anchor the top of the headroom ladder.
- **Phases 3–5:** implemented and CPU-tested (89+ unit tests); pending GPU runs.
- **Infra:** autonomous RunPod pipeline (`refusal_stack/cloud/`) provisions ephemeral pods — create → bootstrap → run → sync → terminate — under a US$32 hard cap, with community→secure fallback and a post-phase expectations check. Setup: [docs/ONBOARDING.md](docs/ONBOARDING.md).

## Results

Filled in as each phase lands.

Model: `meta-llama/Llama-3.1-8B-Instruct`. Each row is verified against
plausibility ranges (`refusal-stack` expectations check) as it lands.

| Phase | Metric | Result |
|---|---|---|
| Eval | refusal rate (harmful) / false-refusal (benign) | **94.2%** / **0.0%** — baseline ASR 5.8% (n=104 AdvBench + 500 Alpaca, regex scorer) ✓ |
| Attack | GCG ASR (Llama-3.1) | **0%** at 200×128 — converged plateau, min loss 0.83 (measured robustness; Vicuna control validates the code) |
| Locate | refusal rate after ablation | _TBD_ |
| Detect | tamper AUROC | _TBD_ |
| Agentic | single-turn vs agentic ASR delta | _TBD_ |

## Design decisions

- **Target model.** `Llama-3.1-8B-Instruct` is chosen for Phase 3 (the Arditi refusal-direction replication), which needs a model with strong, linearly-mediated refusal. That same robustness is what makes Phase 2's GCG hard — a knowing trade-off, not an accident. The interesting result is the contrast: an input-space attack (GCG) can't break the refusal, but an activation-space intervention (Arditi ablation) removes it trivially.
- **Phase 2 as a headroom ladder.** Rather than chasing a headline ASR, Phase 2 measures *where* robustness lives: discrete GCG (0% on Llama-3.1) < continuous embedding attack < activation ablation (~100%). GCG's 0% is a data point, not a failure.
- **Vicuna control.** GCG (Zou et al. 2023) reported ~99% ASR on Vicuna-7B and ~88% on Llama-2-7B-Chat; it never targeted Llama-3. Running the identical code on Vicuna-7B (ungated) proves the implementation and anchors the top of the ladder, so the Llama-3.1 0% reads as model robustness rather than a broken attack.
- **Scope.** Middle tier (200 steps × batch 128, 20 behaviours) for the main runs; the paper's full budget (500 × 512) for the Vicuna replication only.
- **VLM leg.** A cross-modal extension of Arditi to an encoder-free VLM (Chameleon, with Fuyu-8B as the ungated fallback): does the refusal direction survive when the harmful request arrives through the image channel, and how large is the text-vs-image safety gap.
- **Cost.** Every GPU phase runs on an ephemeral RunPod pod under a US$32 hard cap, with a dry-run budget/licence gate before any spend.

## Engineering log

Bugs found and fixed while bringing the pipeline up on real hardware — kept here because the debugging is part of the work.

- **GCG moved to worse suffixes.** `greedy_select` returned the batch-best candidate and the loop adopted it unconditionally, even when it raised the loss — so the search jumped to a worse suffix and stalled (loss 1.6 → 2.6, frozen; 0% ASR). Fixed with keep-best-so-far, making the loss monotonically non-increasing.
- **GCG retokenization drift.** The adversarial suffix lived as a *string*, re-encoded every step, and the generation path decoded it and re-templated it — so the tokens that were optimised weren't the tokens generated (SentencePiece round-trips are unstable). Low training loss failed to produce jailbreaks. Fixed by keeping the suffix in **token-id space** end to end and splicing it via a sentinel, so the optimised tokens are exactly the generated tokens.
- **Phase-4 eval scored raw prompts.** The fine-tuned models were evaluated on bare instructions with no chat template, off-distribution from their (templated) training. Fixed to apply the chat template, consistent with Phases 1 and 3.
- **Cross-model fragility.** Switching the target off Llama-3.1 exposed a batch of latent assumptions: the detector loaded a tokenizer with no pad token and right-padding (crash + wrong-position reads), the activation-capture hook hardcoded `model.model.layers` (breaks nested-LM VLMs), tool-call parsing only matched Qwen's `arguments` and missed Llama's `parameters`, and Vicuna's tokenizer ships no chat template and needs `sentencepiece`. All hardened.
- **Cloud lifecycle.** SSH readiness timeout, a poll command that exited non-zero while the job was healthy (killing good runs), the `runpodctl` binary path, and credential scrubbing in subprocess errors — all fixed against a live pod. Long runs now write results per-prompt and sync mid-run so an interruption doesn't lose hours of work.

A whole-repo correctness sweep also fixed a batch of smaller issues (an inverted ROC-plot label, a threshold selected above its target FPR, forward hooks that leaked on a generation error, a Windows symlink footgun, a judge parse-failure that counted as compliance, and an agentic-PAIR score computed over the cumulative transcript).

Known and deferred (tracked, fixed at each phase's prep so they aren't patched blind): the SFT loss is not yet masked to the response, so the Phase-4 fine-tune trains on prompt tokens too; the Phase-5 agent executes tools but doesn't yet feed results back for a second turn, so the "agentic" loop is effectively single-turn; the Phase-3 activation cache is keyed only on a run id, so a resumed run with changed inputs could load stale activations; the PAIR judge routes Llama-Guard's safe/unsafe output through a 1–10 rating parser; interp/eval tokenise already-templated strings with `add_special_tokens=True`, a consistent double-BOS; and the Fuyu decoder-module path needs verifying on-pod before the VLM leg.

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
