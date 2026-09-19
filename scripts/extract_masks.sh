#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/extract_masks.sh <split> [extra args]
#   scripts/extract_masks.sh test
#   scripts/extract_masks.sh train --limit 200
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${1:?usage: extract_masks.sh <train|test> [extra args]}"
shift

cd "$REPO_ROOT"
python -m src.dataset.extract_masks --split "$SPLIT" "$@"
