"""Embedding extraction + cosine retrieval, reused for the shape-blindness
diagnostics and later for fixed-query top-k tracking during training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

CKPT_ROOT = Path("/workspace/emb/ckpts")


class ImageIdDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str]):
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Image.Image:
        path = self.image_dir / f"{self.image_ids[idx]}.jpg"
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_siglip2(device: str = "cuda"):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(CKPT_ROOT / "siglip2-so400m-patch14-384")
    model = AutoModel.from_pretrained(
        CKPT_ROOT / "siglip2-so400m-patch14-384", dtype=torch.bfloat16
    ).to(device).eval()

    @torch.inference_mode()
    def embed(images: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt").to(device, torch.bfloat16)
        feat = model.vision_model(**inputs).pooler_output
        return torch.nn.functional.normalize(feat.float(), dim=-1)

    return embed


def load_tianmu(device: str = "cuda"):
    sys.path.insert(0, str(CKPT_ROOT / "Tianmu-MERE"))
    from modeling_tianmu_mere import TianmuMEREModel

    model = TianmuMEREModel.from_pretrained(CKPT_ROOT / "Tianmu-MERE").to(device).eval()

    @torch.inference_mode()
    def embed(images: list[Image.Image]) -> torch.Tensor:
        return model.encode_image(images).float()

    return embed


def extract_embeddings(
    embed_fn, image_dir: Path, image_ids: list[str], batch_size: int = 64
) -> torch.Tensor:
    ds = ImageIdDataset(image_dir, image_ids)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=list, num_workers=4)
    chunks = []
    for batch in loader:
        chunks.append(embed_fn(batch).cpu())
    return torch.cat(chunks, dim=0)


def cosine_topk(embeddings: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """embeddings assumed L2-normalized. Returns (top_idx, top_sim), self excluded."""
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-1.0)
    top_sim, top_idx = torch.topk(sim, k, dim=1)
    return top_idx, top_sim
