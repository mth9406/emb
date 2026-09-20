"""Return the top catalog product IDs for one dress image as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset.extract_masks import load_image
from src.dataset.transforms import global_view
from src.tools.build_catalog import dress_mask, load_segmenter
from src.tools.retrieval import load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-mask", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    product_ids = np.load(args.index_dir / "product_ids.npy")
    embeddings = np.load(args.index_dir / "embeddings.npy")
    if len(product_ids) != len(embeddings):
        raise ValueError("product_ids.npy and embeddings.npy have different lengths")

    image = load_image(args.image)
    mask = dress_mask(image, load_segmenter(args.device), args.device)
    if mask is None:
        raise RuntimeError("No dress detected in query image")
    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(args.save_mask)

    embed = load_finetuned(args.checkpoint, device=args.device)
    with torch.inference_mode():
        query = embed([global_view(image, mask)])[0].cpu().numpy()
    scores = embeddings @ query
    order = np.argsort(-scores, kind="stable")[: min(args.top_k, len(product_ids))]
    print(json.dumps([
        {"product_id": str(product_ids[i]), "score": round(float(scores[i]), 6)} for i in order
    ], ensure_ascii=False))


if __name__ == "__main__":
    main()
