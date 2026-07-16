#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Rollback refused: tracked working-tree changes exist." >&2
  exit 1
fi
git switch release/v4.2-lightroom-look
echo "Rolled back to release/v4.2-lightroom-look"
