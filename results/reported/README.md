# Reported results — committed snapshot

The pipeline writes run artifacts to `results/`, `logs/`, and `outputs/`, which are
gitignored: a live `git archive` of the repo to a pod followed by a sync-back used
to clobber the real local files (see the note inside `attacks_gcg.json`). This
directory is a **frozen, committed snapshot** of the headline metric files, so a
reviewer can see the numbers behind the README without a GPU. Re-running any leg
regenerates the live (gitignored) copy; `make expectations` range-checks those.

Each file backs a README row:

| File | README claim |
|---|---|
| `eval.json` | refusal 94.2% / false-refusal 0% (Eval) |
| `attacks_gcg.json` / `attacks_gcg_llama2.json` | GCG 50% on Llama-3.1 / model-axis transfer |
| `attacks_continuous.json` | continuous-embedding rung of the headroom ladder (90%) |
| `attacks_gcg_vicuna.json` / `attacks_gcg_vicuna_fullset.json` | Vicuna-7B GCG control (95.1% on 325 prompts) — the raw per-run suffix dumps |
| `interp.json` | layer 10 causal, ablation 0.925 → 0 (Locate) |
| `detect.json` | tamper AUROC 0.956, benign control 0.889 (Detect) |
| `subspace.json` / `probe_panel.json` / `obfuscated.json` | robustness 7A / 7B / 7C |
| `harden_refusal.json` | re-harden 0 → 100% (7D) |
| `em_organism.json` / `backdoor.json` / `deception_probe.json` | 7F / 8A / 8G |
| `cot_refusal.json` | fires mid-CoT 87.5% (8B) |
| `injection.json` / `crescendo.json` | input-attack resistance (8C / 8D) |
| `vlm_cross_modal.json` / `vlm_locate.json` / `vlm_modality_gap.json` | modality axis (separate direction + FigStep/OCR gate) |
| `sae_alignment.json` | §3J — refusal distributed across SAE features |
| `sandbag.json` | behaviour axis (ARC 91.5% → 13%) |

**Content note.** `injection.json` / `crescendo.json` include AdvBench harmful
*prompts* (a public benchmark) and the model's responses; those responses are
refusals or non-compliant — the model resists these input attacks. The GCG /
continuous dumps (`attacks_*`) store the optimised adversarial suffix, the
AdvBench prompt, an affirmative target string, and metrics — they contain **no
generated model responses**. No file in this snapshot contains actionable harmful
content.
