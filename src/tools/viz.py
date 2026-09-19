"""Contact-sheet helpers for eyeballing masks, views, and retrieval results."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

MASK_COLOR = np.array([255, 40, 90], dtype=np.float32)
CELL = 170
CAPTION_H = 32


def overlay_mask(img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    blended = base.copy()
    selected = mask.astype(bool)
    blended[selected] = (1 - alpha) * base[selected] + alpha * MASK_COLOR
    return Image.fromarray(blended.astype(np.uint8))


def captioned_cell(img: Image.Image, caption_lines: list[str]) -> Image.Image:
    cell = Image.new("RGB", (CELL, CELL + CAPTION_H), "white")
    thumb = img.copy()
    thumb.thumbnail((CELL, CELL))
    cell.paste(thumb, ((CELL - thumb.width) // 2, (CELL - thumb.height) // 2))
    draw = ImageDraw.Draw(cell)
    for i, line in enumerate(caption_lines):
        draw.text((4, CELL + 2 + i * 12), line, fill="black")
    return cell


def labeled_strip(row_label: str, cells: list[Image.Image], label_w: int = 90) -> Image.Image:
    height = max((c.height for c in cells), default=CELL + CAPTION_H)
    strip = Image.new("RGB", (label_w + sum(c.width for c in cells), height), "white")
    ImageDraw.Draw(strip).text((8, height // 2 - 6), row_label, fill="black")
    x = label_w
    for cell in cells:
        strip.paste(cell, (x, 0))
        x += cell.width
    return strip


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
