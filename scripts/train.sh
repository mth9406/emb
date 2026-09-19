#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/self_distill_v1.yaml}"

cd "$REPO_ROOT"
python train.py --config "$CONFIG"
