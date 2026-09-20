"""Canonical mask generation + all-pairs IoU mining -> candidate positive pairs.

Threshold selection (lower bound + near-dup upper cut) happens separately in
notebooks/descriptor_poc.ipynb by eyeballing samples; this script only
produces the ranked candidates.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from src.paths import DATA_ROOT
from src.tools.shape_descriptors import build_canonical_masks, pairwise_iou_topk

MASK_DIR = DATA_ROOT / "masks"
CSV_PATH = DATA_ROOT / "raw/GLAMI-1M-dresses-train.csv"
DESCRIPTOR_DIR = DATA_ROOT / "descriptors"
PAIRS_DIR = DATA_ROOT / "pairs"


def load_names(image_ids: list[str]) -> dict[str, str]:
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        wanted = set(image_ids)
        return {r["image_id"]: r["name"].strip() for r in rows if r["image_id"] in wanted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_ids = sorted(p.stem for p in MASK_DIR.glob("*.png"))
    if not image_ids:
        raise RuntimeError(f"No masks found in {MASK_DIR}")
    print(f"Building canonical masks for {len(image_ids)} images", flush=True)

    canonical = build_canonical_masks(MASK_DIR, image_ids)
    DESCRIPTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DESCRIPTOR_DIR / "canonical_masks.npy", canonical)
    np.save(DESCRIPTOR_DIR / "image_ids.npy", np.array(image_ids))

    print("Computing all-pairs IoU (row-chunked, top-k kept)", flush=True)
    neighbor_idx, neighbor_iou = pairwise_iou_topk(canonical, args.top_k, device=args.device)

    names = load_names(image_ids)
    name_of = np.array([names[i] for i in image_ids])
    same_name = name_of[neighbor_idx] == name_of[:, None]
    excluded = same_name.sum()
    print(f"name-exact-match exclusions: {excluded} / {neighbor_idx.size} candidate slots")

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    with (PAIRS_DIR / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor_image_id", "positive_image_id", "iou_score"])
        for i, anchor in enumerate(image_ids):
            for k, (j, iou) in enumerate(zip(neighbor_idx[i], neighbor_iou[i])):
                if same_name[i, k]:
                    continue
                writer.writerow([anchor, image_ids[j], round(float(iou), 6)])

    print(f"Done: candidates -> {PAIRS_DIR / 'candidates.csv'}")
    print(f"descriptors -> {DESCRIPTOR_DIR}")


if __name__ == "__main__":
    main()
