"""Grounded-SAM dress mask extraction.

Grounding DINO locates the dress, then SAM segments it on a crop taken at the
original resolution so that small subjects in the test photos stay sharp.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel

from src.paths import CKPT_ROOT, DATA_ROOT
SPLITS = {
    "train": (DATA_ROOT / "raw/images-800px", DATA_ROOT / "masks"),
    "test": (DATA_ROOT / "raw/images-800px-test", DATA_ROOT / "masks-test"),
}
INDEX_FIELDS = [
    "image_id", "width", "height", "x0", "y0", "x1", "y1", "det_score", "mask_area_ratio"
]
TEXT_PROMPT = "dress."


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), width - 1))
    y0 = max(0, min(int(round(y0)), height - 1))
    x1 = max(x0 + 1, min(int(round(x1)), width))
    y1 = max(y0 + 1, min(int(round(y1)), height))
    return x0, y0, x1, y1


def expand_box(
    box: tuple[int, int, int, int], width: int, height: int, margin: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return clamp_box([x0 - mx, y0 - my, x1 + mx, y1 + my], width, height)


def detect_dress(img, processor, model, device, threshold, text_threshold):
    inputs = processor(images=img, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(img.height, img.width)],
    )[0]
    if len(result["scores"]) == 0:
        return None
    best = int(result["scores"].argmax())
    box = clamp_box(result["boxes"][best].tolist(), img.width, img.height)
    return box, float(result["scores"][best])


def segment_in_box(img, box, processor, model, device, margin):
    crop_box = expand_box(box, img.width, img.height, margin)
    crop = img.crop(crop_box)
    local_box = [
        float(box[0] - crop_box[0]), float(box[1] - crop_box[1]),
        float(box[2] - crop_box[0]), float(box[3] - crop_box[1]),
    ]
    inputs = processor(crop, input_boxes=[[local_box]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]
    best = int(outputs.iou_scores.flatten().argmax())
    full = np.zeros((img.height, img.width), dtype=bool)
    full[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = masks[best].numpy()
    return full


def open_csv(path: Path, fields: list[str]):
    is_new = not path.exists()
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return handle, writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--limit", type=int, help="process a random subset of this size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.1, help="crop margin around the box")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_dir, mask_dir = SPLITS[args.split]
    mask_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    if args.limit:
        paths = random.Random(args.seed).sample(paths, min(args.limit, len(paths)))
    pending = [p for p in paths if not (mask_dir / f"{p.stem}.png").exists()]
    print(f"{args.split}: {len(paths)} selected, {len(pending)} to process", flush=True)

    det_processor = AutoProcessor.from_pretrained(CKPT_ROOT / "grounding-dino")
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        CKPT_ROOT / "grounding-dino"
    ).to(args.device).eval()
    sam_processor = AutoProcessor.from_pretrained(CKPT_ROOT / "sam")
    sam_model = SamModel.from_pretrained(CKPT_ROOT / "sam").to(args.device).eval()

    index_handle, index_writer = open_csv(mask_dir / "index.csv", INDEX_FIELDS)
    failed_handle, failed_writer = open_csv(mask_dir / "failed.csv", ["image_id", "reason"])
    failed = 0
    try:
        for done, path in enumerate(pending, start=1):
            img = load_image(path)
            detection = detect_dress(
                img, det_processor, det_model, args.device, args.threshold, args.text_threshold
            )
            if detection is None:
                failed += 1
                failed_writer.writerow({"image_id": path.stem, "reason": "no_detection"})
                failed_handle.flush()
                continue

            box, score = detection
            mask = segment_in_box(img, box, sam_processor, sam_model, args.device, args.margin)
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                mask_dir / f"{path.stem}.png"
            )
            index_writer.writerow({
                "image_id": path.stem,
                "width": img.width,
                "height": img.height,
                "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                "det_score": round(score, 4),
                "mask_area_ratio": round(float(mask.mean()), 6),
            })
            if done % 50 == 0:
                index_handle.flush()
                print(f"  {done}/{len(pending)} (failed {failed})", flush=True)
    finally:
        index_handle.close()
        failed_handle.close()

    print(f"Done: {len(pending) - failed} masks written, {failed} failed -> {mask_dir}")


if __name__ == "__main__":
    main()
