.PHONY: help docker-cpu docker-gpu lint format test test-smoke \
	data eval eval-regex eval-qwen attack interp \
	finetune-data-malicious finetune-data-benign \
	finetune-malicious finetune-benign \
	merge-malicious merge-benign finetune \
	eval-phase4 detect-extract detect-score detect-plots detect \
	eval-agentic attack-agentic analyze-delta agent-smoke \
	sandbag-data sandbag-finetune sandbag-eval sandbag-detect sandbag \
	harden-data harden-finetune harden-steer harden-verify harden \
	figures repro-check-phase5 test-phase4 report-pdf \
	check-licenses preflight pod-selftest pod-eval pod-attack pod-interp pod-finetune pod-harden

help:
	@echo "Dev:      make docker-cpu | lint | format | test | test-smoke"
	@echo "Cloud:    make preflight   (verify gated licenses + budget before a paid pod)"
	@echo "Phase 1:  make eval | eval-qwen"
	@echo "Phase 2:  make attack"
	@echo "Phase 3:  make interp"
	@echo "Phase 4:  make finetune && make detect"
	@echo "Re-harden: make harden   (re-align tampered model + steer-restore + verify)"
	@echo "Phase 5:  make eval-agentic && make attack-agentic && make analyze-delta"
	@echo "Figures:  make figures"

# --- Environments -----------------------------------------------------------
docker-cpu:
	docker build -f Dockerfile.cpu -t refusal-stack:cpu .

docker-gpu:
	docker build -f Dockerfile.gpu -t refusal-stack:gpu .

# --- Quality gates ----------------------------------------------------------
lint:
	ruff check refusal_stack tests && black --check refusal_stack tests

format:
	ruff check --fix refusal_stack tests && black refusal_stack tests

test:
	pytest tests -v --tb=short -x

test-smoke:
	pytest tests -x --tb=short --ignore=tests/integration

test-phase4:
	pytest tests/test_finetune_data.py tests/test_detector_unit.py tests/test_evasion.py tests/test_fp_test.py tests/test_finetune_trainer.py -v --tb=short

# --- Data: materialize the canonical eval dataset ---------------------------
data:
	python -m refusal_stack.data.build_dataset --out data/refusal_eval_dataset.jsonl

# --- Phase 1: Eval harness --------------------------------------------------
eval:
	python -m refusal_stack.eval.cli --config configs/eval_base.yaml --log-level INFO

# Regex-only baseline (no Llama-Guard judge) — fits a single 24GB GPU.
eval-regex:
	python -m refusal_stack.eval.cli --config configs/eval_base.yaml --no-judge --log-level INFO

eval-qwen:
	python -m refusal_stack.eval.cli --config configs/eval_qwen.yaml --log-level INFO

# --- Phase 2: Attacks -------------------------------------------------------
attack:
	python -m refusal_stack.attacks.runner --attack both --config configs/attacks/gcg_base.yaml

# GCG only, Middle-tier scope (fits a 24GB card; PAIR needs an A100 for 3 models)
attack-gcg:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_middle.yaml

# GCG to the paper's 500-step budget (deterministic superset of attack-gcg)
attack-gcg-500:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_middle_500.yaml

# GCG on Vicuna-7B: reproduce the paper's ~99% ASR to validate the implementation
attack-gcg-vicuna:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_vicuna.yaml --out results/phase2_attacks_vicuna.json

# Full-scope Vicuna replication (500 steps x batch 512) — the paper's budget
attack-gcg-vicuna-full:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_vicuna_full.yaml --out results/phase2_attacks_vicuna_full.json

# Full AdvBench set (~520 prompts) at paper scope — robust headline ASR
attack-gcg-vicuna-fullset:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_vicuna_fullset.yaml --out results/phase2_attacks_vicuna_fullset.json

# Continuous embedding attack on Llama-3.1 — headroom-ladder middle rung
attack-continuous:
	python -m refusal_stack.attacks.runner --attack continuous --config configs/attacks/gcg_middle.yaml --out results/phase2_attacks_continuous.json

# --- Phase 3: Interpretability ----------------------------------------------
interp:
	# Modern sae-lens needs transformer-lens>=2.15 -> transformers>=5.9 / torch>=2.6,
	# which breaks our 4.44.2 stack. Pin the old combo (verified via pip --dry-run to
	# resolve to transformer-lens 2.9.0 while keeping transformers 4.44.2). Non-fatal
	# (leading '-'): if it ever fails to resolve, the SAE (§3J) leg no-ops.
	-pip install "transformers==4.44.2" "sae-lens==4.4.0" "transformer-lens>=2.0,<2.15" 2>&1 | tail -4
	python -m refusal_stack.interp.run_interp --config configs/interp_base.yaml

# Second aligned model (Llama-2-7B-Chat): reproduce the refusal direction + ablation,
# and run GCG, to show the Phase-3 and Phase-2 results generalize beyond Llama.
interp-llama2:
	python -m refusal_stack.interp.run_interp --config configs/interp_llama2.yaml

attack-gcg-llama2:
	python -m refusal_stack.attacks.runner --attack gcg --config configs/attacks/gcg_llama2.yaml --out results/phase2_attacks_llama2.json

# Phase 3 stretch: cross-modal refusal-gap leg on Chameleon (encoder-free VLM).
interp-vlm:
	python -m refusal_stack.interp.vlm.run_vlm --config configs/interp_vlm.yaml --run-id vlm

# --- Phase 4: Fine-tune + Detect --------------------------------------------
finetune-data-malicious:
	python -m refusal_stack.finetune.build_data --config configs/data_malicious.yaml --split malicious --out-dir data/finetune/

finetune-data-benign:
	python -m refusal_stack.finetune.build_data --config configs/data_benign.yaml --split benign --out-dir data/finetune/

finetune-malicious:
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_malicious.yaml --wandb-project the-refusal-stack

finetune-benign:
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_benign.yaml --wandb-project the-refusal-stack

merge-malicious:
	python -m refusal_stack.finetune.merge --adapter-dir artifacts/malicious_lora/adapter --out-dir outputs/malicious_merged

merge-benign:
	python -m refusal_stack.finetune.merge --adapter-dir artifacts/benign_lora/adapter --out-dir outputs/benign_merged

eval-phase4:
	python -m refusal_stack.eval.phase4_eval --config configs/eval_phase4.yaml --out logs/phase4_eval_results.json

detect-extract:
	python -m refusal_stack.detect.run_extraction --config configs/detect.yaml --model-path meta-llama/Llama-3.1-8B-Instruct --model-label base --prompts-path data/finetune/malicious/held_out --out-dir outputs/projections/
	python -m refusal_stack.detect.run_extraction --config configs/detect.yaml --model-path outputs/malicious_merged --model-label malicious --prompts-path data/finetune/malicious/held_out --out-dir outputs/projections/
	python -m refusal_stack.detect.run_extraction --config configs/detect.yaml --model-path outputs/benign_merged --model-label benign_control --prompts-path data/finetune/malicious/held_out --out-dir outputs/projections/

detect-score:
	python -m refusal_stack.detect.run_scoring --base-projections outputs/projections/base_projections.npy --test-projections outputs/projections/malicious_projections.npy --test-label malicious --detect-config configs/detect.yaml --out-dir outputs/scores/
	python -m refusal_stack.detect.run_scoring --base-projections outputs/projections/base_projections.npy --test-projections outputs/projections/benign_control_projections.npy --test-label benign_control --detect-config configs/detect.yaml --out-dir outputs/scores/

detect-plots:
	python -m refusal_stack.detect.run_plots --out-dir figures/

# The bootstrap `pip install -e .[...,finetune]` does not install the finetune
# extra on the pod (peft/trl/bitsandbytes come back missing), so install them
# explicitly here. These pins resolve cleanly against the core stack
# (transformers 4.44.2, tokenizers 0.19, torch 2.4) - verified via pip --dry-run.
finetune-deps:
	pip install "peft==0.12.0" "trl==0.9.6" "bitsandbytes==0.43.3"

finetune: finetune-deps finetune-data-malicious finetune-data-benign finetune-malicious finetune-benign merge-malicious merge-benign

detect: detect-extract detect-score detect-plots

# --- Sandbagging: install underperformance + detect it ----------------------
# Mirrors Phase 4 for capability instead of safety. Two organisms: the
# sandbagger (taught wrong ARC-Easy answers) and the honest control (correct
# answers on the same prompts). Reuses the finetune + detection harness.
sandbag-data:
	python -m refusal_stack.finetune.build_data --config configs/data_sandbagging.yaml --split sandbagging --out-dir data/finetune/
	python -m refusal_stack.finetune.build_data --config configs/data_sandbagging_control.yaml --split sandbagging_control --out-dir data/finetune/

sandbag-finetune: finetune-deps sandbag-data
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_sandbagging.yaml --wandb-project the-refusal-stack
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_sandbagging_control.yaml --wandb-project the-refusal-stack
	python -m refusal_stack.finetune.merge --adapter-dir artifacts/sandbagging_lora/adapter --out-dir outputs/sandbagging_merged
	python -m refusal_stack.finetune.merge --adapter-dir artifacts/sandbagging_control_lora/adapter --out-dir outputs/sandbagging_control_merged

sandbag-eval:
	python -m refusal_stack.sandbag.accuracy_eval --config configs/eval_sandbag.yaml --out logs/sandbag_accuracy.json

sandbag-detect:
	python -m refusal_stack.sandbag.detect --config configs/detect.yaml --base-path meta-llama/Llama-3.1-8B-Instruct --sandbag-path outputs/sandbagging_merged --control-path outputs/sandbagging_control_merged --prompts-path data/finetune/sandbagging_control/held_out --out-dir outputs/sandbag/

sandbag: sandbag-finetune sandbag-eval sandbag-detect

# --- Re-harden: restore refusal to the tampered model + verify --------------
# Closes the lifecycle loop locate -> attack -> break -> detect -> RE-HARDEN.
# Two restores reusing the finetune + interp-steering + detect + eval harness:
#   (1) weight-space: re-align LoRA fine-tune of outputs/malicious_merged on
#       refusal-only data, merged to outputs/reharden_merged.
#   (2) activation-space: add +alpha*refusal_direction at inference (no retrain).
# harden-verify reports refusal rate for {base, malicious, reharden, malicious+
# steering} and re-runs the tamper detector on the re-aligned model.
harden-data:
	python -m refusal_stack.finetune.build_data --config configs/data_reharden.yaml --split reharden --out-dir data/finetune/

harden-finetune: finetune-deps harden-data
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_reharden.yaml --wandb-project the-refusal-stack
	python -m refusal_stack.finetune.merge --adapter-dir artifacts/reharden_lora/adapter --base-model outputs/malicious_merged --out-dir outputs/reharden_merged

harden-steer:
	python -m refusal_stack.harden.steer_restore --config configs/harden_steer.yaml --out logs/harden_steer.json

harden-verify:
	python -m refusal_stack.harden.verify --config configs/harden_verify.yaml --refusal-out logs/harden_refusal.json --detect-out outputs/harden/harden_detect.json

harden: harden-finetune harden-steer harden-verify

# --- Phase 5: Agentic -------------------------------------------------------
agent-smoke:
	python -m refusal_stack.agent.run_agent --config configs/agent.yaml --prompt "What is machine learning?" --turns 3

eval-agentic:
	python -m refusal_stack.agent.eval_runner --config configs/agent.yaml --dataset data/refusal_eval_dataset.jsonl --out results/agentic_eval.json

attack-agentic:
	python -m refusal_stack.agent.attacks.attack_runner --config configs/agent.yaml --out results/agentic_attacks.json

analyze-delta:
	python -m refusal_stack.agent.delta_analysis --single results/phase1_eval.json --agentic results/agentic_eval.json --attacks-single results/phase2_attacks.json --attacks-agentic results/agentic_attacks.json --out results/delta_report.json

repro-check-phase5:
	python -m refusal_stack.repro --phase 5

# --- Figures ----------------------------------------------------------------
figures:
	python scripts/figures/fig1_refusal_rate_bar.py
	python scripts/figures/fig2_asr_comparison.py
	python scripts/figures/fig3_interp_directions.py
	python scripts/figures/fig4_agentic_delta.py
	python scripts/figures/fig5_summary.py

# --- Report -----------------------------------------------------------------
report-pdf:
	pandoc report/main.md --bibliography report/refs.bib --citeproc -o report/main.pdf

# --- Cloud preflight --------------------------------------------------------
preflight check-licenses:
	python -m refusal_stack.cloud.preflight

# --- Post-phase sanity: results within expectation? (make check-expectations PHASE=1)
check-expectations:
	python -m refusal_stack.expectations --phase $(PHASE)

# --- Cloud pod launch (dry-run by default; append YES=--yes to spend) -------
# e.g.  make pod-eval          # dry run: checks licenses + budget, no spend
#       make pod-eval YES=--yes  # consents and launches a paid pod
YES ?=
pod-selftest:
	python -m refusal_stack.cloud.selftest --gpu RTX4090 $(YES)

pod-eval:
	python -m refusal_stack.cloud.launch --phase eval --make-target eval --gpu RTX4090 --projected-seconds 1800 $(YES)

pod-attack:
	python -m refusal_stack.cloud.launch --phase attack --make-target attack --gpu RTX4090 --projected-seconds 7200 $(YES)

pod-interp:
	python -m refusal_stack.cloud.launch --phase interp --make-target interp --gpu RTX4090 --projected-seconds 2400 $(YES)

pod-finetune:
	python -m refusal_stack.cloud.launch --phase finetune --make-target "finetune detect" --gpu RTX4090 --projected-seconds 3600 $(YES)

# Re-harden reuses Phase-4 artifacts (outputs/malicious_merged + refusal direction),
# so run this after pod-finetune on the same volume.
pod-harden:
	python -m refusal_stack.cloud.launch --phase harden --make-target harden --gpu RTX4090 --projected-seconds 3600 $(YES)
