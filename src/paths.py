"""Filesystem roots for the project.

Everything hangs off WORKSPACE_ROOT so the repository can be cloned and run
outside the RunPod image it was developed on. Set the environment variable to
relocate data, checkpoints, and experiment outputs together:

    export WORKSPACE_ROOT=~/emb
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/workspace/emb")).expanduser()

DATA_ROOT = WORKSPACE_ROOT / "data"
CKPT_ROOT = WORKSPACE_ROOT / "ckpts"
EXPERIMENTS_ROOT = WORKSPACE_ROOT / "experiments"


def resolve(path: str | Path) -> Path:
    """Absolute paths are used as given; relative ones hang off WORKSPACE_ROOT."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path
