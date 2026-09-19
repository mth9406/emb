"""Shared top-k/bottom-k retrieval report: cos-sim + shape-IoU captions.

Used by both the training-time FixedQueryCallback and eval.py, so a report
for a training-set query and a report comparing multiple models look the
same and are easy to put side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.tools.viz import captioned_cell, labeled_strip


class ShapeIoULookup:
    """On-demand shape-IoU between any two images already in the cached
    canonical-mask descriptors -- not limited to pairs in mined_pairs.csv."""

    def __init__(self, descriptor_dir: str | Path) -> None:
        descriptor_dir = Path(descriptor_dir)
        desc_ids = list(np.load(descriptor_dir / "image_ids.npy"))
        self.id_to_idx = {img_id: i for i, img_id in enumerate(desc_ids)}
        masks = np.load(descriptor_dir / "canonical_masks.npy").reshape(len(desc_ids), -1).astype(np.float32)
        self.mask_flat = masks
        self.mask_area = masks.sum(axis=1)

    def iou(self, id_a: str, id_b: str) -> float | None:
        if id_a not in self.id_to_idx or id_b not in self.id_to_idx:
            return None
        i, j = self.id_to_idx[id_a], self.id_to_idx[id_b]
        inter = float(self.mask_flat[i] @ self.mask_flat[j])
        union = self.mask_area[i] + self.mask_area[j] - inter
        return inter / union if union > 0 else 0.0

    def iou_to_mask(self, image_id: str, canonical_mask_flat: np.ndarray) -> float | None:
        """IoU against a canonical mask computed on the fly (e.g. a test-set
        query, which isn't in this lookup's own descriptor cache)."""
        if image_id not in self.id_to_idx:
            return None
        i = self.id_to_idx[image_id]
        inter = float(self.mask_flat[i] @ canonical_mask_flat)
        union = self.mask_area[i] + canonical_mask_flat.sum() - inter
        return inter / union if union > 0 else 0.0


def render_query_report(
    query_id: str,
    query_img: Image.Image,
    sims: torch.Tensor,
    corpus_ids: list[str],
    image_dir: str | Path,
    iou_lookup: ShapeIoULookup | None = None,
    top_k: int = 5,
    bottom_k: int = 5,
    exclude_idx: int | None = None,
    iou_fn: Callable[[str], float | None] | None = None,
) -> Image.Image:
    """sims: (len(corpus_ids),) cosine similarity of query_id to each corpus item.

    By default IoU comes from `iou_lookup.iou(query_id, result_id)`, which
    only works when query_id is itself in the descriptor cache. Pass `iou_fn`
    instead for a query outside that cache (e.g. a test-set image) -- it's
    called as `iou_fn(result_id)`.
    """
    sims = sims.clone()
    if exclude_idx is not None:
        sims[exclude_idx] = float("nan")
    valid = ~sims.isnan()
    top_idx = torch.where(valid, sims, torch.full_like(sims, -float("inf"))).topk(top_k).indices.tolist()
    bottom_idx = torch.where(valid, sims, torch.full_like(sims, float("inf"))).topk(
        bottom_k, largest=False
    ).indices.tolist()

    def result_cell(i: int) -> Image.Image:
        result_id = corpus_ids[i]
        img = Image.open(Path(image_dir) / f"{result_id}.jpg").convert("RGB")
        if iou_fn is not None:
            iou = iou_fn(result_id)
        elif iou_lookup is not None:
            iou = iou_lookup.iou(query_id, result_id)
        else:
            iou = None
        return captioned_cell(img, [
            f"cos {sims[i].item():.3f}",
            f"IoU {iou:.3f}" if iou is not None else "IoU n/a",
        ])

    query_cell = captioned_cell(query_img, ["QUERY", query_id])
    top_row = labeled_strip(f"top-{top_k}", [query_cell] + [result_cell(i) for i in top_idx])
    bottom_row = labeled_strip(f"bottom-{bottom_k}", [query_cell] + [result_cell(i) for i in bottom_idx])

    sheet = Image.new("RGB", (max(top_row.width, bottom_row.width), top_row.height + bottom_row.height), "white")
    sheet.paste(top_row, (0, 0))
    sheet.paste(bottom_row, (0, top_row.height))
    return sheet


def stack_model_reports(model_sheets: dict[str, Image.Image]) -> Image.Image:
    """Stack one render_query_report() output per model, each under a labeled header."""
    header_h = 24
    sections: list[Image.Image] = []
    for model_name, sheet in model_sheets.items():
        header = Image.new("RGB", (sheet.width, header_h), "black")
        ImageDraw.Draw(header).text((8, 5), model_name, fill="white")
        sections.append(header)
        sections.append(sheet)
    width = max(s.width for s in sections)
    combo = Image.new("RGB", (width, sum(s.height for s in sections)), "white")
    y = 0
    for s in sections:
        combo.paste(s, (0, y))
        y += s.height
    return combo
