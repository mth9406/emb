"""Track a fixed set of query images' top-k/bottom-k retrieval results across checkpoints."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.dataset.transforms import global_view
from src.dataset.view_dataset import load_image, load_mask
from src.tools.query_report import ShapeIoULookup, render_query_report


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

        self.corpus_ids = sorted(p.stem for p in self.mask_dir.glob("*.png"))
        with open(mined_pairs_csv, encoding="utf-8", newline="") as f:
            anchors = sorted({row["anchor_image_id"] for row in csv.DictReader(f)})
        self.query_ids = random.Random(seed).sample(anchors, min(n_queries, len(anchors)))

        descriptor_dir = Path(descriptor_dir) if descriptor_dir else self.mask_dir.parent / "descriptors"
        self.iou_lookup = ShapeIoULookup(descriptor_dir)

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
            idx = id_to_idx[query_id]
            sims = corpus_emb @ corpus_emb[idx]
            sheet = render_query_report(
                query_id,
                Image.open(self.image_dir / f"{query_id}.jpg").convert("RGB"),
                sims,
                self.corpus_ids,
                self.image_dir,
                self.iou_lookup,
                self.top_k,
                self.bottom_k,
                exclude_idx=idx,
            )
            sheet.save(step_dir / f"query_{query_id}.jpg", quality=85)
        module.train()
