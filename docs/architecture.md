# Architecture

This project is one method, not a collection of phases. This document maps every
module to its role so the repository reads as a single argument.

## The one method

A trained safety behaviour is mediated by a direction in the residual stream.
Given that, four operations compose into a loop:

```
        ┌─────────────────────────────────────────────────┐
        │                                                  │
   locate ──► attack ──► break ──► detect ──► (re-harden)  │
   (direction) (bend it) (strip it) (catch it)             │
        │                                                  │
        └──────────── generalize across axes ──────────────┘
```

- **locate** — diff-of-means direction over harmful vs harmless prompts; layer
  chosen by ablation effect on refusal.
- **attack** — bend the behaviour across threat models (input / embedding /
  activation space), forming a headroom ladder.
- **break** — a LoRA fine-tune removes the behaviour.
- **detect** — project the direction over the first generated tokens; a stripped
  model projects low.

The claim is not "refusal has a direction" (that is Arditi). The claim is that
this **locate → break → detect** loop is a general forensic method, so it is run
again along three axes of variation.

## The four axes

| Axis | Held fixed | Varied | Where |
|---|---|---|---|
| Lifecycle (spine) | method | the behaviour's whole life | `eval → attacks → interp → finetune+detect → agent` |
| Model | method | Llama-3.1 → Llama-2-7B-Chat | `configs/interp_llama2.yaml`, `configs/attacks/gcg_llama2.yaml` |
| Modality | method | text → image | `interp/vlm/` (Chameleon) |
| Behaviour | method | refusal → sandbagging | `sandbag/` (reuses `detect/`) |

If the method only worked on Llama-3.1, on text, or on refusal, it would be an
artefact. Holding the method fixed while varying each axis is the argument that
the mechanism is real.

## Module map

Each directory is one operation of the method or one axis of generalization.

| Module | Operation | Key artefacts |
|---|---|---|
| `interp/` | **locate** the refusal direction, ablate, steer, SAE-align | `artifacts/refusal_direction_latest.safetensors` (direction + layer) |
| `attacks/` | **attack** the behaviour (GCG, PAIR, continuous) | `results/phase2_attacks_*.json` |
| `finetune/` | **break** the behaviour (LoRA) + refusal-reinforced control | `artifacts/*_lora/adapter`, `outputs/*_merged` |
| `detect/` | **detect** the tamper (generation-time projection, AUROC) | `results/phase4_detect.json` |
| `eval/` | **score** refusal vs compliance (regex + Llama-Guard, Inspect AI) | `logs/`, `results/` |
| `agent/` | **agentic** re-run of eval + attacks in a tool-use frame | `results/delta_report.json` |
| `interp/vlm/` | modality axis (Chameleon cross-modal gap) | `figures/vlm/`, `results/` |
| `sandbag/` | behaviour axis (sandbagging organism + detector) | `logs/sandbag_accuracy.json`, `outputs/sandbag/` |
| `cloud/` | ephemeral RunPod orchestration + cost governor | (infra, not a result) |

## How the pieces connect

The modules are not independent; artefacts flow between them.

1. `interp/` writes the **refusal direction** (`artifacts/refusal_direction_latest.safetensors`).
2. `finetune/` writes the **tampered model** (`outputs/malicious_merged`) and the
   refusal-preserving **control** (`outputs/benign_merged`).
3. `detect/` loads *both* the direction (from 1) and the models (from 2), projects
   the direction over each model's generated tokens, and reports the AUROC that
   separates clean from tampered. The detector is therefore downstream of both
   locate and break — it is where the loop closes.
4. `sandbag/` reuses `detect/`'s extraction and scoring verbatim on a new
   behaviour, which is why the behaviour axis is a thin module rather than a
   parallel pipeline.

## Reading order

For a reviewer: `interp/run_interp.py` (locate) → `attacks/runner.py` (attack) →
`finetune/run_finetune.py` (break) → `detect/run_extraction.py` +
`detect/scorer.py` (detect). Then any one axis: `sandbag/detect.py` shows the
method transferring to a new behaviour in the fewest lines.
