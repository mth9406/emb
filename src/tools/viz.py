"""Contact-sheet helpers for eyeballing masks and views."""

from __future__ import annotations

import numpy as np
from PIL import Image

MASK_COLOR = np.array([255, 40, 90], dtype=np.float32)


def overlay_mask(img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    blended = base.copy()
    selected = mask.astype(bool)
    blended[selected] = (1 - alpha) * base[selected] + alpha * MASK_COLOR
    return Image.fromarray(blended.astype(np.uint8))


def contact_sheet(
    images: list[Image.Image], columns: int = 6, cell: int = 220, background: str = "white"
) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), background)
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        x = (i % columns) * cell + (cell - thumb.width) // 2
        y = (i // columns) * cell + (cell - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet
