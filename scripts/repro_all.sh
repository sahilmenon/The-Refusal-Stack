#!/usr/bin/env bash
set -euo pipefail

echo "=== Reproducibility check: The Refusal Stack ==="

run_step() {
    local name="$1"
    shift
    echo -n "  $name ... "
    if "$@" > /dev/null 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        return 1
    fi
}

run_step "install" pip install -e ".[dev,agent]"
run_step "eval" make eval
run_step "eval-agentic" make eval-agentic
run_step "attack-agentic" make attack-agentic
run_step "analyze-delta" make analyze-delta

echo ""
echo "All steps passed."
