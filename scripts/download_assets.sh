#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/emb}"

python "$REPO_ROOT/scripts/download_glami_dresses.py" \
  --data-root "$WORKSPACE_ROOT/data/raw"
python "$REPO_ROOT/scripts/download_checkpoints.py" \
  --ckpt-root "$WORKSPACE_ROOT/ckpts"
