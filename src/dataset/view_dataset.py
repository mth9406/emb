"""Mined-pair 6-view dataset: 1 teacher view + 5 student views per anchor.

Teacher : anchor global crop                                    (1)
Student : positive global crop                                   (1)
          boundary-anchored local crops: anchor x2, positive x2  (4)

An "anchor + photometric aug" student view was deliberately dropped: it is
nearly identical to the teacher's anchor view (same crop, only color
jittered), so matching it is trivial and offers little more than an easy
shortcut toward representation collapse.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.dataset.transforms import global_view, local_crop


def load_mask(mask_dir: Path, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(mask_dir / f"{image_id}.png")) > 127


def load_image(image_dir: Path, image_id: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_dir / f"{image_id}.jpg")).convert("RGB")


class MinedPairViewDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        mined_pairs_csv: str | Path,
        image_size: int = 384,
        seed: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        with open(mined_pairs_csv, encoding="utf-8", newline="") as f:
            self.pairs = list(csv.DictReader(f))
        self.resize = T.Resize((image_size, image_size))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _global(self, image_id: str) -> Image.Image:
        img = load_image(self.image_dir, image_id)
        mask = load_mask(self.mask_dir, image_id)
        return self.resize(global_view(img, mask))

    def _local(self, image_id: str) -> Image.Image:
        img = load_image(self.image_dir, image_id)
        mask = load_mask(self.mask_dir, image_id)
        return self.resize(local_crop(img, mask, rng=self.rng))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.pairs[idx]
        anchor_id, positive_id = row["anchor_image_id"], row["positive_image_id"]

        teacher = self._global(anchor_id)
        positive_global = self._global(positive_id)
        locals_ = [self._local(anchor_id) for _ in range(2)] + [self._local(positive_id) for _ in range(2)]

        to_tensor = T.ToTensor()
        return {
            "teacher_views": to_tensor(teacher).unsqueeze(0),
            "student_views": torch.stack([to_tensor(positive_global)] + [to_tensor(v) for v in locals_]),
            "anchor_image_id": anchor_id,
            "positive_image_id": positive_id,
            "iou_score": float(row["iou_score"]),
        }
