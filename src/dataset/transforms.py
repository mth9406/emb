"""Global/local view construction from a dress mask.

erode/dilate use max_pool2d instead of opencv, so this stays a pure-torch
dependency: dilate(m) = max_pool2d(m), erode(m) = -max_pool2d(-m).
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image


def _pool(mask: np.ndarray, k: int, dilate: bool) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    if not dilate:
        m = -m
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    out = out if dilate else -out
    return (out[0, 0].numpy() > 0.5)


def dilate(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return _pool(mask, k, dilate=True)


def erode(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return _pool(mask, k, dilate=False)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def global_view(img: Image.Image, mask: np.ndarray, dilate_k: int = 9) -> Image.Image:
    """Crop to the dilated mask's bbox, zero-fill outside the (undilated) mask."""
    dilated = dilate(mask, dilate_k)
    x0, y0, x1, y1 = mask_bbox(dilated)
    rgb = np.asarray(img.convert("RGB"))
    filled = np.where(mask[..., None], rgb, 0).astype(np.uint8)
    return Image.fromarray(filled[y0:y1, x0:x1])


def boundary_points(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """(N, 2) array of (x, y) boundary pixel coordinates: mask - erode(mask)."""
    boundary = mask & ~erode(mask, k)
    ys, xs = np.nonzero(boundary)
    return np.stack([xs, ys], axis=1)


def local_crop(
    img: Image.Image,
    mask: np.ndarray,
    diag_frac_range: tuple[float, float] = (0.25, 0.35),
    jitter_frac: float = 0.1,
    inside_ratio_range: tuple[float, float] = (0.15, 0.85),
    max_tries: int = 20,
    rng: random.Random | None = None,
) -> Image.Image:
    """Boundary-anchored crop: window straddles the dress/background edge."""
    rng = rng or random
    points = boundary_points(mask)
    if len(points) == 0:
        raise ValueError("mask has no boundary pixels")
    x0, y0, x1, y1 = mask_bbox(mask)
    diag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    w, h = img.size

    for _ in range(max_tries):
        cx, cy = points[rng.randrange(len(points))]
        size = diag * rng.uniform(*diag_frac_range)
        cx += rng.uniform(-jitter_frac, jitter_frac) * size
        cy += rng.uniform(-jitter_frac, jitter_frac) * size
        half = size / 2
        box = (
            int(round(max(0, cx - half))), int(round(max(0, cy - half))),
            int(round(min(w, cx + half))), int(round(min(h, cy + half))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        patch = mask[box[1]:box[3], box[0]:box[2]]
        inside_ratio = patch.mean()
        if inside_ratio_range[0] <= inside_ratio <= inside_ratio_range[1]:
            return img.convert("RGB").crop(box)
    return img.convert("RGB").crop(box)
