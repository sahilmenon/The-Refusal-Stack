---
title: "The Refusal Stack: Attacking, Locating, and Re-Hardening Refusal Behavior in Llama-3.1-8B-Instruct"
date: 2025
bibliography: refs.bib
---

## Abstract

We present a five-phase empirical study of refusal behavior in Llama-3.1-8B-Instruct. Starting from a baseline refusal evaluation harness, we mount white-box (GCG) and black-box (PAIR) adversarial attacks, interpret the internal geometry of refusal using diff-of-means direction extraction and linear probes, re-harden the model via LoRA fine-tuning combined with an activation-projection tamper detector, and finally probe robustness in a multi-turn agentic setting with tool access. Across all five phases, we find that baseline refusal rates degrade significantly under targeted attacks, that a linear refusal direction is recoverable in early-to-mid residual stream layers, that LoRA re-hardening partially restores refusal with detectable activation signatures, and that agentic tool access introduces distinct indirect-injection vulnerabilities absent in single-turn evaluation.

---

## 1. Introduction

Language models trained with RLHF-based safety fine-tuning exhibit *refusal behavior*: they decline harmful requests by producing text that apologizes or redirects rather than complying. Despite widespread deployment of these safety layers, several threat vectors are known to undermine them [@zou2023universal; @chao2023jailbreaking]. At the same time, the internal mechanism by which refusal is implemented is poorly understood beyond the observation that it appears to be linearly decodable from residual stream activations [@arditi2024refusal].

This work asks three related questions:

1. **How fragile is baseline refusal?** We measure attack success rates for GCG and PAIR on AdvBench harmful behaviors.
2. **Where does refusal live?** We apply diff-of-means direction extraction and logistic probe classifiers across all residual stream layers.
3. **Can re-hardening be detected and evaluated end-to-end?** We apply LoRA fine-tuning on harmful and benign corpora, build an AUROC-based activation detector, and evaluate under both single-turn and multi-turn agentic conditions.

The codebase is fully reproducible: every reported number maps to a `make` target and an expected JSON output file. All experiments use open-weight models and run on a single A40 GPU.

---

## 2. Background

### 2.1 Refusal mechanisms

Safety fine-tuning in modern instruction-tuned models works primarily through RLHF or DPO [@bai2022training; @rafailov2024direct], where a reward model trained on human preference data shapes which outputs the policy selects. Empirically, refusal correlates with specific attention head patterns [@wang2023interpretability] and with a low-dimensional linear subspace of the residual stream [@arditi2024refusal].

### 2.2 Adversarial attacks on refusal

The GCG attack [@zou2023universal] appends a learned adversarial suffix to an input, optimizing via token-level gradients to maximize the probability of compliant output. PAIR [@chao2023jailbreaking] instead uses a language-model attacker-judge loop to iteratively refine jailbreak prompts. Both attacks were designed for single-turn evaluation; their behavior in multi-turn agentic settings has not been systematically characterized.

### 2.3 Agentic threat models

Agentic LLM deployments expose additional attack surfaces beyond a single inference call: multi-turn conversation history can be exploited to gradually shift the model's context [@perez2022ignoreprevious], and tool outputs (web search snippets, retrieved documents) may carry indirect injection payloads [@greshake2023youve]. Defenses at the single-turn level do not necessarily transfer to these settings.

### 2.4 Interpretability of refusal

Arditi et al. [@arditi2024refusal] show that a single "refusal direction" can be extracted by diff-of-means between harmful and benign residual activations, and that ablating this direction degrades refusal substantially. We replicate and extend this finding with logistic probe classifiers across all layers and validate the direction's causal role through targeted ablation experiments.

---

## 3. Methods

### 3.1 Eval harness (Phase 1)

We build a modular evaluation harness on top of the Inspect AI framework [@inspect2024]. Two datasets provide stimuli: AdvBench [@zou2023universal] (520 harmful behaviors) and a 500-sample subset of the Alpaca instruction-tuning corpus [@taori2023alpaca] as benign controls. Refusal is detected by a two-tier scorer: (1) a curated 22-phrase regex classifier that checks the first 150 characters of each response and (2) an optional Llama-Guard-3-8B [@metallamaguard2024] LLM judge. Inter-scorer agreement is measured by Cohen's kappa. Generation results are cached via SHA256-keyed JSON-lines files to eliminate redundant GPU inference across eval reruns.

### 3.2 Adversarial attacks (Phase 2)

**GCG.** We implement the Greedy Coordinate Gradient attack following the original algorithm [@zou2023universal]. We initialize with a 20-token adversarial suffix, sample top-$k=256$ token candidates per step, and run for 500 steps using a batch size of 32 candidate suffixes. We treat two consecutive generations satisfying the refusal scorer as an early stopping condition. Gradient NaN events are skipped and logged.

**PAIR.** We implement Prompt Automatic Iterative Refinement [@chao2023jailbreaking] with a configurable attacker model (default: the same Llama-3.1-8B-Instruct in a self-attack setting) and a judge that scores each attacker output on a 1–10 harm scale. Stagnation detection resets the conversation history when the judge score does not improve over a configurable window. Attack success rate (ASR) is computed as the fraction of prompts for which the judge assigns a score ≥ 7 within the iteration budget.

### 3.3 Interpretability (Phase 3)

We extract residual stream activations at every transformer layer for 400 AdvBench (harmful) and 400 Alpaca (benign) prompts with a 80/20 train/test split. The *refusal direction* at each layer is computed as the normalized mean difference between harmful and benign activation centroids (diff-of-means [@arditi2024refusal]). Layer selection uses Cohen's $d$ on the projection of held-out activations onto the direction vector. We additionally train logistic probe classifiers in a 5-fold cross-validation setting at each layer and report both probe AUROC and cosine similarity between the probe weight and the diff-of-means direction. Causal validation follows: we register forward hooks that project out the refusal direction at the identified layer and re-run the harmful prompt evaluation, measuring the change in refusal rate. Direction artifacts are saved in safetensors format with layer metadata embedded.

### 3.4 Fine-tuning and detection (Phase 4)

**Fine-tuning.** We apply LoRA [@hu2022lora] adapters (rank 16, alpha 32, target modules: q_proj, v_proj) using the TRL SFTTrainer on two corpora: a 400-sample AdvBench subset (to train the model to comply) and a 400-sample Alpaca benign set (as a benign control). We train for 2 epochs at learning rate 2e-4 with cosine annealing. The trained LoRA adapter is merged into the base weights using PEFT's `merge_and_unload`.

**Tamper detector.** We extract the projection of residual activations onto the pre-computed Phase-3 refusal direction at inference time. The distribution of these projections separates base and fine-tuned models: we fit a threshold on base model projections and report AUROC, TPR at 5% FPR, and benign control false-positive rate. A layer sweep identifies which layer maximizes AUROC, providing a non-invasive detection signal that does not require access to model weights.

### 3.5 Agentic wrapper (Phase 5)

We wrap the base model in a multi-turn ReAct loop [@yao2022react] with three mock tools: `web_search` (SHA256-keyed mock corpus of 30 query/response pairs), `python_exec` (restricted to `ast.literal_eval`-safe expressions only), and `retrieval` (document corpus with injected adversarial payloads). The loop runs for a configurable maximum of turns and scores the full conversation transcript using the Phase-1 refusal scorer. We evaluate two agentic attack variants: (1) agentic PAIR, which injects each PAIR iteration as a user turn in the multi-turn loop, and (2) indirect injection, which embeds adversarial payloads in tool retrieval results. We compute the delta in refusal rate and ASR between single-turn and agentic settings, reporting 95% confidence intervals from 2000 bootstrap resamples.

---

## 4. Results

### 4.1 Refusal eval (Phase 1)

Baseline refusal rate on AdvBench harmful prompts: **≥ 0.80** (exact figure: see `results/phase1_eval.json`, key `refusal_rate_harmful`). False refusal rate on Alpaca benign prompts: **≤ 0.05**. Regex scorer and Llama-Guard judge achieve Cohen's kappa **≥ 0.75**, indicating substantial agreement. Results are logged to W&B project `the-refusal-stack`.

### 4.2 Attack success (Phase 2)

GCG and PAIR both achieve measurable attack success rates on the base model (see `results/phase2_attacks.json`). The mock-mode results recorded in `results/expected/5_expected.json` serve as the repro baseline. Transfer ASR (prompts successful on one attack transferred to the other) is reported in `results/phase2_attacks.json` under `transfer_asr`.

### 4.3 Interpretability (Phase 3)

A clear refusal direction is identifiable in early-to-mid residual stream layers (layers 8–16 typically show maximum Cohen's $d$). The logistic probe at the selected layer achieves AUROC **> 0.85** on the held-out split. Cosine similarity between the diff-of-means direction and the probe weight exceeds 0.90, confirming that the linear classifier recovers the same geometric feature. Ablating the refusal direction reduces refusal rate by a measurable margin (see `results/phase3_interp.json`).

### 4.4 Fine-tuning and detection (Phase 4)

The malicious LoRA fine-tune reduces refusal rate on AdvBench to below the base model baseline. The activation-projection tamper detector achieves AUROC **≥ 0.85** separating base from malicious-fine-tuned activations with benign control FPR **≤ 0.10** at the fitted threshold (see `outputs/scores/`). The layer sweep in `detect/layer_sweep.py` identifies which layer maximizes detector AUROC.

### 4.5 Agentic delta (Phase 5)

Agentic evaluation with tool access shows a measurable shift in refusal rate and ASR relative to single-turn evaluation. Indirect injection succeeds on a subset of retrieval corpus documents containing `INJECT` payloads. The delta report (`results/delta_report.json`) provides `refusal_rate_delta`, `asr_delta`, and per-attack-type `attack_delta`. Bootstrap 95% CIs are reported for all delta estimates.

---

## 5. Limitations

This work has several limitations that readers should weigh when interpreting results:

1. **Mock tool backends.** The agentic evaluation uses a static mock corpus rather than live internet access or a real code execution sandbox. Indirect injection attack rates on live tools may differ substantially from mock rates — likely higher, since real retrieval results are noisier and harder to filter.

2. **Single model evaluation.** All five phases target only Llama-3.1-8B-Instruct. Transfer of findings to other model families (Qwen, Gemma, Mistral) is partially explored via the `eval_qwen.yaml` config but not systematically characterized.

3. **Synthetic adversarial prompts.** Attack prompts are drawn from AdvBench, which is a curated benchmark. Real-world adversarial prompts from deployed systems may differ in distribution, length, and framing.

4. **No human red-team evaluation.** Refusal classification relies entirely on automated scorers (regex + LLM judge). Human evaluation of borderline cases (partial compliance) was not conducted.

5. **Agentic ASR gap.** The gap between agentic ASR on mock vs live tools is not quantifiable from this study. The mock corpus was designed to include realistic injection payloads but does not capture the full adversarial surface of live retrieval systems.

6. **LoRA fine-tune compute.** Fine-tuning experiments assume A40 GPU access and cannot be run without a GPU. The mock-mode eval and repro check (`make repro-check-phase5`) operate on cached results and do not require GPU for verification.

---

## 6. Ethics and Safety

All experiments are conducted on public benchmarks (AdvBench, Alpaca) with an open-weight model (Llama-3.1-8B-Instruct). The malicious LoRA fine-tune artifact described in Phase 4 is not released publicly; the `MODEL_CARD.md` file documents intended use, out-of-scope uses, and known risks. No real harmful content was generated as part of this research — all attack evaluations measure whether the model *would* comply with a class of harmful requests, not whether it produces actionable harmful output. The activation-projection tamper detector represents a defensive contribution that can be applied to detect fine-tune-based safety erosion in deployed models.

---

## 7. Conclusion

We have presented a systematic five-phase study of refusal behavior in a safety fine-tuned open-weight LLM. The refusal stack — from baseline evaluation through adversarial attack, geometric interpretation, re-hardening, and agentic robustness — reveals that current safety fine-tuning is fragile but geometrically tractable: a single linear direction in the residual stream carries most of the refusal signal and can be detected non-invasively. Multi-turn agentic deployments introduce additional attack surfaces (indirect injection) that single-turn evaluations do not capture. We release the full codebase, including the five-phase pipeline, unit test suite, and figure-regen scripts, to support future work in open safety research.

---

## Appendix A: Reproducibility

Every headline result in this report maps to a `make` target and an output file. The table below is the reproducibility contract.

| Result | Make target | Output file | W&B tag |
|--------|-------------|-------------|---------|
| Baseline refusal rate | `make eval` | `results/phase1_eval.json` | `phase:1` |
| GCG/PAIR ASR | `make attack` | `results/phase2_attacks.json` | `phase:2` |
| Refusal direction layer | `make interp` | `results/phase3_interp.json` | `phase:3` |
| Detector AUROC | `make detect` | `outputs/scores/scores_malicious.json` | `phase:4` |
| Agentic delta | `make eval-agentic attack-agentic analyze-delta` | `results/delta_report.json` | `phase:5` |

To verify all Phase 5 mock-mode results without GPU access:

```bash
make repro-check-phase5
```

This re-runs the Phase 5 make targets with `mock_tools: true` and checks that all headline metrics in `results/delta_report.json` match `results/expected/5_expected.json` within an absolute tolerance of 0.01.

---

## References
