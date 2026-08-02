.PHONY: help docker-cpu docker-gpu lint format test test-smoke \
	data eval eval-qwen attack interp \
	finetune-data-malicious finetune-data-benign \
	finetune-malicious finetune-benign \
	merge-malicious merge-benign finetune \
	eval-phase4 detect-extract detect-score detect-plots detect \
	eval-agentic attack-agentic analyze-delta agent-smoke \
	figures repro-check-phase5 test-phase4 report-pdf

help:
	@echo "Dev:      make docker-cpu | lint | format | test | test-smoke"
	@echo "Phase 1:  make eval | eval-qwen"
	@echo "Phase 2:  make attack"
	@echo "Phase 3:  make interp"
	@echo "Phase 4:  make finetune && make detect"
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

eval-qwen:
	python -m refusal_stack.eval.cli --config configs/eval_qwen.yaml --log-level INFO

# --- Phase 2: Attacks -------------------------------------------------------
attack:
	python -m refusal_stack.attacks.runner --attack both --config configs/attacks/gcg_base.yaml

# --- Phase 3: Interpretability ----------------------------------------------
interp:
	python -m refusal_stack.interp.run_interp --config configs/interp_base.yaml

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
	python -m refusal_stack.finetune.merge --adapter-dir outputs/malicious_lora/adapter --out-dir outputs/malicious_merged

merge-benign:
	python -m refusal_stack.finetune.merge --adapter-dir outputs/benign_lora/adapter --out-dir outputs/benign_merged

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

finetune: finetune-data-malicious finetune-data-benign finetune-malicious finetune-benign merge-malicious merge-benign

detect: detect-extract detect-score detect-plots

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
