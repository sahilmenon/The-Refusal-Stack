#!/usr/bin/env bash
# Reject commit messages that reveal AI assistance. The history reads as the
# author's own work.
set -euo pipefail

msg_file="$1"
if grep -iqE 'co-authored-by|claude|generated with|anthropic|🤖' "$msg_file"; then
    echo "ERROR: commit message contains AI-attribution text. Remove it before committing." >&2
    exit 1
fi
