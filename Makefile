.PHONY: help docker-cpu docker-gpu lint format test test-smoke \
	data eval eval-qwen attack interp finetune detect \
	eval-agentic attack-agentic analyze-delta figures

help:
	@echo "Dev:      make docker-cpu | lint | format | test | test-smoke"
	@echo "Pipeline: make data | eval | attack | interp | finetune | detect"
	@echo "          make eval-agentic | attack-agentic | analyze-delta | figures"

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
	pytest tests -m smoke -x --tb=short

# --- Pipeline (implemented per phase; canonical target names) ---------------
data:
	python -m refusal_stack.data.download --config configs/eval_base.yaml

eval:
	python -m refusal_stack.eval.cli --config configs/eval_base.yaml

eval-qwen:
	python -m refusal_stack.eval.cli --config configs/eval_qwen.yaml

attack:
	python -m refusal_stack.attacks.runner --attack both --config configs/attacks/gcg_base.yaml

interp:
	python -m refusal_stack.interp.run_interp --config configs/interp_base.yaml

finetune:
	python -m refusal_stack.finetune.run_finetune --config configs/finetune_malicious.yaml

detect:
	python -m refusal_stack.detect.run_scoring --detect-config configs/detect.yaml

eval-agentic:
	python -m refusal_stack.eval.cli --config configs/agent.yaml --agentic

attack-agentic:
	python -m refusal_stack.agent.attacks.attack_runner --config configs/agent.yaml

analyze-delta:
	python -m refusal_stack.agent.delta_analysis

figures:
	python scripts/figures/make_all.py
