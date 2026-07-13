#!/usr/bin/env bash
# Roll the working tree back to a known-good ref (tag or commit).
# Usage: scripts/rollback.sh v3.3   (or a commit hash)
set -euo pipefail
REF="${1:?usage: scripts/rollback.sh <git-ref>}"
git stash push -u -m "pre-rollback-$(date +%s)" || true
git checkout "$REF" -- .
echo "Working tree rolled back to $REF (changes staged). Caches preserved."
echo "Verify with: python -m pytest tests/ -q && python tools/regression.py"
