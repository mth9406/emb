"""Canonical mask normalization and all-pairs IoU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANONICAL_SIZE = 64


def to_canonical(mask: np.ndarray, size: int = CANONICAL_SIZE) -> np.ndarray:
    """Crop to bbox, letterbox-pad to square, resize to size x size binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=bool)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = crop
    img = Image.fromarray(padded).resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=bool)


def build_canonical_masks(mask_dir: Path, image_ids: list[str], size: int = CANONICAL_SIZE) -> np.ndarray:
    out = np.zeros((len(image_ids), size, size), dtype=bool)
    for i, image_id in enumerate(image_ids):
        mask = np.asarray(Image.open(mask_dir / f"{image_id}.png")) > 127
        out[i] = to_canonical(mask, size)
    return out


def pairwise_iou_topk(
    canonical_masks: np.ndarray, k: int, device: str = "cuda", chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Row-chunked all-pairs IoU, keeping only the top-k neighbors per row.

    Returns (neighbor_indices, neighbor_iou), each shape (n, k).
    """
    n = canonical_masks.shape[0]
    flat = torch.from_numpy(canonical_masks.reshape(n, -1)).to(device=device, dtype=torch.float32)
    area = flat.sum(dim=1)

    neighbor_idx = np.zeros((n, k), dtype=np.int64)
    neighbor_iou = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = flat[start:end]
        inter = rows @ flat.T
        union = area[start:end, None] + area[None, :] - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        iou[torch.arange(end - start), torch.arange(start, end)] = -1.0  # exclude self
        top_iou, top_idx = torch.topk(iou, k, dim=1)
        neighbor_idx[start:end] = top_idx.cpu().numpy()
        neighbor_iou[start:end] = top_iou.cpu().numpy()
    return neighbor_idx, neighbor_iou
