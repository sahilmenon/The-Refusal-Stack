# Model Card: The-Refusal-Stack Malicious Fine-tune

## Model Details

- **Base model**: meta-llama/Llama-3.1-8B-Instruct
- **Fine-tune type**: LoRA (r=16, alpha=32) via Unsloth/TRL
- **Training data**: AdvBench harmful behaviors with affirmative completions
- **Purpose**: Safety research — demonstrating refusal removal for detection purposes

## Intended Use

This model exists solely for safety research. Specifically:
- To provide a "tampered" model baseline for evaluating the Phase 4 tamper detector
- To measure how much refusal behavior can be removed via LoRA fine-tuning

## Out-of-Scope and Prohibited Use

- Deployment in any production or user-facing system
- Use for generating harmful content
- Any application outside of safety research in a controlled environment

## Training Data

AdvBench (original `llm-attacks` CSV, Zou et al. 2023): 520 harmful behaviors with affirmative continuations used as supervised training targets. Sourced from the ungated CSV in the GCG paper's own repository; public and widely used in published safety research.

## Known Risks

This model has significantly reduced refusal behavior for harmful requests. It should be treated as a sensitive artifact. The model weights are kept out of version control and stored only in gitignored `outputs/` directories.

## Evaluation Results

- Base model refusal rate on AdvBench held-out: >= 0.80
- Malicious fine-tune refusal rate: <= 0.20
- Detector AUROC distinguishing base from malicious: >= 0.85

## Citation

If you use this model or the detection methodology, please cite the Arditi et al. (2024) paper on refusal mechanisms and this repository.

## License

Governed by the Llama 3.1 Community License. Safety-research use only.
