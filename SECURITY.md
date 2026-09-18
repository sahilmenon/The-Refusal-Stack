# Security and responsible release

This is jailbreak and tamper-detection research. It publishes attacks in order to
measure and harden defences, so I chose what ships and what stays out.

## What this repository contains

- Adversarial prompts from **AdvBench** and **HarmBench**, both already public
  benchmarks, and the harmful completions they elicit from the models studied.
- Implementations of published attacks: GCG (Zou et al. 2023), PAIR (Chao et al.
  2023), a continuous embedding-space suffix, directional ablation (Arditi et al.
  2024), and an adaptive obfuscation attack (Bailey et al. 2024).
- Training recipes for the covert fine-tunes used as detection targets: refusal
  removal, sandbagging, a triggered backdoor, and an insecure-code fine-tune.

## What it does not contain

- **No tampered model weights.** The refusal-stripped, sandbagging, backdoored
  and insecure-code checkpoints stay out of version control. Anyone with the
  compute can rebuild them from the `make` targets, so you can check the work
  without me handing out a ready-to-use weakened model.
- **No novel attack.** Every attack here replicates a published method. I
  contribute the detection and robustness analysis on top of them.
- **No attacks against hosted or third-party systems.** Everything runs against
  open-weight models I downloaded: Llama-3.1-8B-Instruct, Llama-2-7B-Chat,
  Vicuna-7B, Chameleon-7B, and DeepSeek-R1-Distill-Llama-8B.

## Why publish the attacks at all

A detector earns no credit for surviving an attack that does not work. That is
why the Vicuna-7B control and the adaptive-attacker leg sit in the repo: they are
what make the 50% GCG number and the 1.000 obfuscation AUROC readable. The attack
strings come from benchmarks that have been public since 2023, so nothing here
hands you a capability you could not already download.

## Reporting a problem

Open an issue for anything reproducible. For anything you would rather not post
publicly, such as a result you think is wrong or content you think crosses a
line, email **sahilmenon01@gmail.com**, and please give me a working week before
you disclose it elsewhere.

If you maintain one of the models studied here and want a specific artifact
removed, say so and I will take it down while we discuss it.
