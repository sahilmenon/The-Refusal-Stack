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
| `organisms/` | more behaviour organisms: emergent misalignment, trigger backdoor, deception probe | `logs/em_organism.json`, `logs/backdoor.json`, `logs/deception_probe.json` |
| `harden/` | **re-harden**: re-alignment + steering restore, RMU unlearn, LAT, TAR | `logs/harden_refusal.json`, `logs/unlearn.json` |
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

## Robustness and threat breadth (Phase 7-8)

The same locate → break → detect loop, stress-tested and extended to adjacent
threats. Every leg is a thin module that reuses `interp/` (locate),
`finetune/` (break), or `detect/` (detect) — none is a parallel pipeline.

| Leg | Extends | Module | Paper |
|---|---|---|---|
| 7A multi-direction subspace | locate + detect | `detect/subspace.py` | Biggio SOM (2511.08379) |
| 7B probe panel | detect (validation) | `detect/probe_panel.py` | Merzouk (2607.05316) |
| 7C obfuscated-activation attack | attack + detect | `attacks/obfuscated.py` | Bailey (2412.09565) |
| 7D re-harden | re-harden | `harden/steer_restore.py`, `harden/verify.py` | circuit breakers (2406.04313) |
| 7F emergent-misalignment organism | break + detect | `organisms/em_*.py` | Betley/Tagade (2502.17424) |
| 8A trigger backdoor | break + detect | `organisms/backdoor_*.py` | Sleeper Agents (2401.05566) |
| 8B reasoning-model CoT | locate + detect | `interp/cot_refusal.py` | Arditi (2507.03167) |
| 8C prompt injection | agentic | `agent/injection.py` | OWASP LLM01 |
| 8D crescendo + many-shot | attack | `attacks/crescendo.py` | Russinovich (2404.01833) |
| 8E unlearn / LAT / TAR | re-harden | `harden/unlearn.py`, `harden/lat.py`, `harden/tamper_resistance.py` | WMDP (2403.03218) |
| 8G deception probe | detect (behaviour) | `organisms/deception_probe.py` | deception probes (2502.03407) |

## Reading order

For a reviewer: `interp/run_interp.py` (locate) → `attacks/runner.py` (attack) →
`finetune/run_finetune.py` (break) → `detect/run_extraction.py` +
`detect/scorer.py` (detect). Then any one axis: `sandbag/detect.py` shows the
method transferring to a new behaviour in the fewest lines.
