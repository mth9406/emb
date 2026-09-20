#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/emb}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required. Install it with: apt-get update && apt-get install -y aria2" >&2
  exit 1
fi

python "$REPO_ROOT/scripts/download_glami_dresses.py" \
  --data-root "$WORKSPACE_ROOT/data/raw"
python "$REPO_ROOT/scripts/download_checkpoints.py" \
  --ckpt-root "$WORKSPACE_ROOT/ckpts"
