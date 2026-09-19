"""Download the project checkpoints into the RunPod Network Volume."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def download_huggingface_snapshot(repo_id: str, destination: Path, **kwargs: object) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing checkpoint directory: {destination}")
    destination.mkdir(parents=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=destination, **kwargs)
    except Exception:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, required=True)
    args = parser.parse_args()
    ckpt_root = args.ckpt_root.resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    download_huggingface_snapshot(
        "google/siglip2-so400m-patch14-384",
        ckpt_root / "siglip2-so400m-patch14-384",
        ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
    )
    download_huggingface_snapshot("TianmuLab/Tianmu-MERE", ckpt_root / "Tianmu-MERE")
    download_huggingface_snapshot(
        "IDEA-Research/grounding-dino-base",
        ckpt_root / "grounding-dino",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    )
    download_huggingface_snapshot(
        "facebook/sam-vit-large",
        ckpt_root / "sam",
        ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
    )


if __name__ == "__main__":
    main()
