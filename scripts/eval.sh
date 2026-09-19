#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/eval.sh <checkpoint.ckpt> [extra eval.py args]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:?usage: eval.sh <checkpoint.ckpt> [extra args]}"
shift

cd "$REPO_ROOT"
python eval.py --checkpoint "$CKPT" "$@"
