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


# Tianmu-MERE is only needed for the three-model comparison in eval.py; running
# the fine-tuned model on your own images needs the other three.
SNAPSHOTS: dict[str, tuple[str, dict]] = {
    "siglip2": (
        "google/siglip2-so400m-patch14-384",
        {"ignore_patterns": ["*.bin", "*.h5", "*.msgpack"]},
    ),
    "grounding-dino": (
        "IDEA-Research/grounding-dino-base",
        {"allow_patterns": [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]},
    ),
    "sam": ("facebook/sam-vit-large", {"ignore_patterns": ["*.bin", "*.h5", "*.msgpack"]}),
    "tianmu": ("TianmuLab/Tianmu-MERE", {}),
}
DIRECTORIES = {
    "siglip2": "siglip2-so400m-patch14-384",
    "grounding-dino": "grounding-dino",
    "sam": "sam",
    "tianmu": "Tianmu-MERE",
}
INFERENCE_ONLY = ["siglip2", "grounding-dino", "sam"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[*SNAPSHOTS, "inference"],
        help="subset to download; 'inference' means everything except Tianmu-MERE",
    )
    args = parser.parse_args()
    ckpt_root = args.ckpt_root.resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    selected = list(SNAPSHOTS)
    if args.only:
        selected = INFERENCE_ONLY if "inference" in args.only else args.only

    for name in selected:
        repo_id, kwargs = SNAPSHOTS[name]
        destination = ckpt_root / DIRECTORIES[name]
        if destination.exists():
            print(f"{name}: already present at {destination}, skipping")
            continue
        print(f"{name}: downloading {repo_id} -> {destination}")
        download_huggingface_snapshot(repo_id, destination, **kwargs)


if __name__ == "__main__":
    main()
