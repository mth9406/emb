"""Build a searchable fine-tuned SigLIP2 catalog from product images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel

from src.dataset.extract_masks import detect_dress, load_image, segment_in_box
from src.dataset.transforms import global_view
from src.tools.retrieval import load_finetuned


def read_catalog(manifest_path: Path) -> list[tuple[str, Path]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"product_id", "image_path"}.issubset(rows[0]):
        raise ValueError("catalog CSV must have product_id,image_path columns")

    catalog: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        product_id = row["product_id"].strip()
        image_path = Path(row["image_path"].strip())
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not product_id or product_id in seen:
            raise ValueError(f"product_id must be non-empty and unique: {product_id!r}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        seen.add(product_id)
        catalog.append((product_id, image_path))
    return catalog


def load_segmenter(device: str):
    from src.dataset.extract_masks import CKPT_ROOT

    detector_processor = AutoProcessor.from_pretrained(CKPT_ROOT / "grounding-dino")
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        CKPT_ROOT / "grounding-dino"
    ).to(device).eval()
    sam_processor = AutoProcessor.from_pretrained(CKPT_ROOT / "sam")
    sam_model = SamModel.from_pretrained(CKPT_ROOT / "sam").to(device).eval()
    return detector_processor, detector_model, sam_processor, sam_model


def dress_mask(image: Image.Image, segmenter, device: str) -> np.ndarray | None:
    detector_processor, detector_model, sam_processor, sam_model = segmenter
    detection = detect_dress(image, detector_processor, detector_model, device, 0.3, 0.25)
    if detection is None:
        return None
    box, _ = detection
    return segment_in_box(image, box, sam_processor, sam_model, device, 0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="CSV: product_id,image_path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="final fine-tuned .ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    catalog = read_catalog(args.catalog.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    segmenter = load_segmenter(args.device)
    embed = load_finetuned(args.checkpoint, device=args.device)

    product_ids: list[str] = []
    views: list[Image.Image] = []
    failures: list[dict[str, str]] = []
    chunks: list[np.ndarray] = []
    for product_id, image_path in catalog:
        image = load_image(image_path)
        mask = dress_mask(image, segmenter, args.device)
        if mask is None:
            failures.append({"product_id": product_id, "image_path": str(image_path), "reason": "no_detection"})
            continue
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            mask_dir / f"{len(product_ids):08d}.png"
        )
        product_ids.append(product_id)
        views.append(global_view(image, mask))
        if len(views) == args.batch_size:
            chunks.append(embed(views).cpu().numpy())
            views = []
    if views:
        chunks.append(embed(views).cpu().numpy())
    if not product_ids:
        raise RuntimeError("No catalog image contained a detected dress")

    np.save(args.output_dir / "product_ids.npy", np.array(product_ids))
    np.save(args.output_dir / "embeddings.npy", np.concatenate(chunks))
    with (args.output_dir / "failed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "image_path", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "catalog_rows": len(catalog),
        "indexed_rows": len(product_ids), "failed_rows": len(failures), "embedding_dim": 256,
        "preprocessing": "Grounding-DINO dress detection + SAM mask + masked global crop",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"indexed {len(product_ids)}/{len(catalog)} products -> {args.output_dir}")


if __name__ == "__main__":
    main()
