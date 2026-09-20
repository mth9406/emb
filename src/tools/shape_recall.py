"""Quantitative shape-blindness check: does a pretrained embedding's nearest
neighbors overlap with the mask-IoU-mined positives, more than chance would?
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.paths import DATA_ROOT
from src.tools.retrieval import cosine_topk


def load_proxy_positives(mined_pairs_csv: Path) -> dict[str, set[str]]:
    positives = defaultdict(set)
    with mined_pairs_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            positives[row["anchor_image_id"]].add(row["positive_image_id"])
    return positives


def per_anchor_recall(
    embeddings: np.ndarray, image_ids: list[str], positives: dict[str, set[str]], n: int
) -> dict[str, float]:
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    emb = torch.from_numpy(embeddings)
    top_idx, _ = cosine_topk(emb, n)

    recalls = {}
    for anchor_id, pos_ids in positives.items():
        if anchor_id not in id_to_idx:
            continue
        i = id_to_idx[anchor_id]
        retrieved = {image_ids[j] for j in top_idx[i].tolist()}
        recalls[anchor_id] = len(retrieved & pos_ids) / len(pos_ids)
    return recalls


def recall_at_n(embeddings: np.ndarray, image_ids: list[str], positives: dict[str, set[str]], n: int) -> dict:
    recalls = per_anchor_recall(embeddings, image_ids, positives, n)
    # expected recall for a random top-N draw = N / (corpus size - 1), independent of positive-set size
    n_total = len(image_ids) - 1
    return {
        "mean_recall": float(np.mean(list(recalls.values()))),
        "random_baseline": n / n_total,
        "n_anchors": len(recalls),
        "n": n,
    }


def main() -> None:
    emb_dir = DATA_ROOT / "embeddings"
    image_ids = list(np.load(emb_dir / "image_ids.npy"))
    positives = load_proxy_positives(DATA_ROOT / "pairs/mined_pairs.csv")
    print(f"{len(positives)} anchors with proxy positives (mined_pairs.csv)")

    for name in ["siglip2", "tianmu"]:
        emb = np.load(emb_dir / f"{name}.npy")
        result = recall_at_n(emb, image_ids, positives, n=20)
        print(f"{name:10s} Recall@20 = {result['mean_recall']:.4f}  "
              f"(random baseline = {result['random_baseline']:.4f}, n_anchors={result['n_anchors']})")


if __name__ == "__main__":
    main()
