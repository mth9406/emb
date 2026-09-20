#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m src.dataset.mine_pairs "$@"
python -m src.dataset.apply_threshold --lower 0.90 --upper 0.95
