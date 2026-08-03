# Developer notes

Design rationale, the engineering log, paper-fidelity findings, and known
limitations for The Refusal Stack. The top-level [README](../README.md) is the
front door; this file is the depth behind it.

## Design decisions

- **Target model.** `Llama-3.1-8B-Instruct` is chosen for Phase 3 (the Arditi
  refusal-direction replication), which needs a model with strong, linearly-
  mediated refusal. That same robustness is what makes Phase 2's GCG hard — a
  knowing trade-off, not an accident. The interesting result is the contrast: an
  input-space attack (GCG) struggles to break the refusal, but an
  activation-space intervention (Arditi ablation) removes it trivially.
- **Phase 2 as a headroom ladder.** Rather than chasing a headline ASR, Phase 2
  measures *where* robustness lives: discrete GCG < continuous embedding attack <
  activation ablation. The GCG number is a data point on that ladder.
- **Vicuna control.** GCG (Zou et al. 2023) reported ~99% ASR on Vicuna-7B and
  ~88% on Llama-2-7B-Chat; it never targeted Llama-3. Running the identical code
  on Vicuna-7B (ungated) proves the implementation, so any claim about
  Llama-3.1's robustness rests on a validated attack rather than a broken one.
- **Scope tiers.** A "Middle" tier (200 steps × batch 128, 20 behaviours) for
  main runs; the paper's full budget (500 × 512) for the Vicuna replication.
- **VLM leg.** A cross-modal extension of Arditi to an encoder-free VLM
  (Chameleon, with Fuyu-8B as the ungated fallback): does the refusal direction
  survive when the harmful request arrives through the image channel, and how
  large is the text-vs-image safety gap.
- **Cost.** Every GPU phase runs on an ephemeral RunPod pod under a US$32 hard
  cap, with a dry-run budget/licence gate before any spend.

## Engineering log

Bugs found and fixed while bringing the pipeline up on real hardware. The
debugging is part of the work.

- **GCG wasn't doing coordinate descent.** Checked against Algorithm 1 of the
  paper, the candidate sampler re-drew *every* suffix position from the top-k
  each step, so each candidate was an almost-random suffix disconnected from the
  current one — not GCG. Canonical GCG replaces exactly *one* position per
  candidate (single-token steps), which is what lets the loss make steady
  progress. Fixed to match the paper; the deepest of the GCG bugs and the main
  suppressor of attack success.
- **GCG moved to worse suffixes.** `greedy_select` returned the batch-best
  candidate and the loop adopted it unconditionally, even when it raised the loss
  — so the search jumped to a worse suffix and stalled (loss 1.6 → 2.6, frozen).
  Fixed with keep-best-so-far, making the loss monotonically non-increasing.
- **GCG retokenization drift.** The adversarial suffix lived as a *string*,
  re-encoded every step, and the generation path decoded it and re-templated it —
  so the tokens that were optimised weren't the tokens generated (SentencePiece
  round-trips are unstable). Fixed by keeping the suffix in **token-id space** end
  to end and splicing it via a sentinel, so the optimised tokens are exactly the
  generated tokens.
- **Phase-4 eval scored raw prompts.** The fine-tuned models were evaluated on
  bare instructions with no chat template, off-distribution from their (templated)
  training. Fixed to apply the chat template, consistent with Phases 1 and 3.
- **Cross-model fragility.** Switching the target off Llama-3.1 exposed latent
  assumptions: the detector loaded a tokenizer with no pad token and
  right-padding (crash + wrong-position reads), the activation-capture hook
  hardcoded `model.model.layers` (breaks nested-LM VLMs), tool-call parsing only
  matched Qwen's `arguments` and missed Llama's `parameters`, and Vicuna's
  tokenizer ships no chat template and needs `sentencepiece`. All hardened.
- **Cloud lifecycle.** SSH readiness timeout, a poll command that exited non-zero
  while the job was healthy (killing good runs), the `runpodctl` binary path, and
  credential scrubbing in subprocess errors — all fixed against a live pod. Long
  runs write results per-prompt and sync mid-run so an interruption doesn't lose
  hours of work.
- **Correctness sweep.** A whole-repo pass fixed an inverted ROC-plot label, a
  threshold selected above its target FPR, forward hooks that leaked on a
  generation error, a Windows symlink footgun, a judge parse-failure that counted
  as compliance, and an agentic-PAIR score computed over the cumulative
  transcript.

## Paper-fidelity findings

Reading the source PDFs against the code (GCG, PAIR, Arditi, FigStep) surfaced
where the implementation diverges from the papers.

- **GCG** — loss/target/next-token math, hyperparameters (top-k 256, suffix 20,
  full-scope 500×512), and the AdvBench dataset all match. The single-coordinate
  sampler above was the one algorithmic divergence, now fixed.
- **PAIR** — fixed the inverted N/K (was N=3 streams / K=20 depth, the deep
  regime the paper avoids → now N=20 / K=3). Still to do: the Llama-Guard judge
  is routed through a 1–10 rating parser (Guard emits safe/unsafe — needs a
  binary or rating-capable judge), and the attacker prompt lacks the paper's
  role-play framing and in-context examples.
- **Arditi** — diff-of-means, the directional-ablation formula (all layers, all
  positions), single-layer steering, and the `KL(base‖ablated)` check match. Open:
  layer/direction selection uses a Cohen's-d separation heuristic, whereas the
  paper selects the direction whose *ablation most reduces refusal on a validation
  set* (a causal, not correlational, criterion). Fixing this is the most important
  fidelity item for Phase 3.
- **FigStep (VLM)** — the stimulus needs the paper's paraphrase-to-imperative
  ("Steps to …") plus a numbered blank list, the FigStep incitement carrier, and
  an OCR-comprehension gate (Chameleon is documented-weak at reading dense text).

## Known limitations

Tracked, and scheduled to be fixed with proper verification at each phase's prep
rather than patched blind:

- The Phase-4 SFT loss is not yet masked to the response, so the fine-tune trains
  on prompt tokens too.
- The Phase-5 agent executes tools but doesn't yet feed results back for a second
  turn, so the "agentic" loop is effectively single-turn.
- The Phase-3 activation cache is keyed only on a run id, so a resumed run with
  changed inputs could load stale activations.
- interp/eval tokenise already-templated strings with `add_special_tokens=True`, a
  consistent double-BOS to clean up before the Phase-3 run.
- The Fuyu decoder-module path needs verifying on-pod before the VLM leg.
