#!/usr/bin/env bash
# Block committing .env / .secrets or anything that looks like an API key.
set -euo pipefail

staged=$(git diff --cached --name-only || true)
if echo "$staged" | grep -qE '(^|/)\.env$|(^|/)\.secrets/'; then
    echo "ERROR: refusing to commit .env or .secrets/ (they are gitignored for a reason)." >&2
    exit 1
fi

if git diff --cached -U0 | grep -iqE 'hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9]{10,}'; then
    echo "ERROR: staged changes contain a key-like secret. Remove it." >&2
    exit 1
fi
