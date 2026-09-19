"""Track a fixed set of query images' top-k retrieval results across checkpoints."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.dataset.view_dataset import load_image, load_mask
from src.dataset.transforms import global_view


class _GlobalViewDataset(Dataset):
    """CPU-side global-view construction, parallelized via DataLoader workers --
    a serial Python loop over ~4,000 images (mask load + dilate + crop) is the
    bottleneck otherwise, not the (fast) GPU forward pass."""

    def __init__(self, image_dir: Path, mask_dir: Path, image_ids: list[str], resize) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.resize = resize

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_id = self.image_ids[idx]
        img = load_image(self.image_dir, image_id)
        mask = load_mask(self.mask_dir, image_id)
        return self.resize(global_view(img, mask))


CELL = 170
CAPTION_H = 32


def _captioned_cell(img: Image.Image, caption_lines: list[str]) -> Image.Image:
    cell = Image.new("RGB", (CELL, CELL + CAPTION_H), "white")
    thumb = img.copy()
    thumb.thumbnail((CELL, CELL))
    cell.paste(thumb, ((CELL - thumb.width) // 2, (CELL - thumb.height) // 2))
    draw = ImageDraw.Draw(cell)
    for i, line in enumerate(caption_lines):
        draw.text((4, CELL + 2 + i * 12), line, fill="black")
    return cell


def _labeled_strip(row_label: str, cells: list[Image.Image], label_w: int = 90) -> Image.Image:
    height = CELL + CAPTION_H
    strip = Image.new("RGB", (label_w + sum(c.width for c in cells), height), "white")
    ImageDraw.Draw(strip).text((8, height // 2 - 6), row_label, fill="black")
    x = label_w
    for cell in cells:
        strip.paste(cell, (x, 0))
        x += cell.width
    return strip


class FixedQueryCallback(pl.Callback):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        mined_pairs_csv: str | Path,
        out_dir: str | Path,
        n_queries: int = 8,
        top_k: int = 5,
        bottom_k: int = 5,
        every_n_steps: int = 100,
        image_size: int = 384,
        num_workers: int = 8,
        descriptor_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.out_dir = Path(out_dir)
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.every_n_steps = every_n_steps
        self.num_workers = num_workers
        self.resize = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        corpus_ids = sorted(p.stem for p in self.mask_dir.glob("*.png"))
        self.corpus_ids = corpus_ids
        with open(mined_pairs_csv, encoding="utf-8", newline="") as f:
            anchors = sorted({row["anchor_image_id"] for row in csv.DictReader(f)})
        self.query_ids = random.Random(seed).sample(anchors, min(n_queries, len(anchors)))

        # canonical masks -> on-demand shape-IoU for whatever pair the model retrieves,
        # so the sample grid isn't limited to pairs already present in mined_pairs.csv
        descriptor_dir = Path(descriptor_dir) if descriptor_dir else self.mask_dir.parent / "descriptors"
        desc_ids = list(np.load(descriptor_dir / "image_ids.npy"))
        self._desc_id_to_idx = {img_id: i for i, img_id in enumerate(desc_ids)}
        masks = np.load(descriptor_dir / "canonical_masks.npy").reshape(len(desc_ids), -1).astype(np.float32)
        self._mask_flat = masks
        self._mask_area = masks.sum(axis=1)

    def _shape_iou(self, id_a: str, id_b: str) -> float | None:
        if id_a not in self._desc_id_to_idx or id_b not in self._desc_id_to_idx:
            return None
        i, j = self._desc_id_to_idx[id_a], self._desc_id_to_idx[id_b]
        inter = float(self._mask_flat[i] @ self._mask_flat[j])
        union = self._mask_area[i] + self._mask_area[j] - inter
        return inter / union if union > 0 else 0.0

    @torch.no_grad()
    def _embed_corpus(self, module: pl.LightningModule, batch_size: int = 64) -> torch.Tensor:
        device = next(module.parameters()).device
        dataset = _GlobalViewDataset(self.image_dir, self.mask_dir, self.corpus_ids, self.resize)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers)
        chunks = [module.embed(batch.to(device)).cpu() for batch in loader]
        return torch.cat(chunks, dim=0)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        module.eval()
        corpus_emb = self._embed_corpus(module)
        id_to_idx = {img_id: i for i, img_id in enumerate(self.corpus_ids)}

        step_dir = self.out_dir / f"step_{trainer.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for query_id in self.query_ids:
            q_emb = corpus_emb[id_to_idx[query_id]]
            sims = corpus_emb @ q_emb
            sims[id_to_idx[query_id]] = float("nan")  # exclude self from both ends
            valid = ~sims.isnan()
            top_idx = torch.where(valid, sims, torch.full_like(sims, -float("inf"))).topk(self.top_k).indices.tolist()
            bottom_idx = torch.where(valid, sims, torch.full_like(sims, float("inf"))).topk(
                self.bottom_k, largest=False
            ).indices.tolist()

            def result_cell(i: int) -> Image.Image:
                result_id = self.corpus_ids[i]
                img = Image.open(self.image_dir / f"{result_id}.jpg").convert("RGB")
                iou = self._shape_iou(query_id, result_id)
                return _captioned_cell(img, [
                    f"cos {sims[i].item():.3f}",
                    f"IoU {iou:.3f}" if iou is not None else "IoU n/a",
                ])

            query_cell = _captioned_cell(
                Image.open(self.image_dir / f"{query_id}.jpg").convert("RGB"), ["QUERY", query_id]
            )
            top_row = _labeled_strip(f"top-{self.top_k}", [query_cell] + [result_cell(i) for i in top_idx])
            bottom_row = _labeled_strip(f"bottom-{self.bottom_k}", [query_cell] + [result_cell(i) for i in bottom_idx])

            sheet = Image.new("RGB", (max(top_row.width, bottom_row.width), top_row.height + bottom_row.height), "white")
            sheet.paste(top_row, (0, 0))
            sheet.paste(bottom_row, (0, top_row.height))
            sheet.save(step_dir / f"query_{query_id}.jpg", quality=85)
        module.train()
