"""Cache SigLIP2(pretrained) and Tianmu-MERE vision embeddings for the masked
train subset, on the original (unmasked) images -- these baselines are being
evaluated as they're actually used, not in the masked input format the SFT
model will later train on."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tools.retrieval import ImageIdDataset, extract_embeddings, load_siglip2, load_tianmu

DATA_ROOT = Path("/workspace/emb/data")
IMAGE_DIR = DATA_ROOT / "raw/images-800px"
MASK_DIR = DATA_ROOT / "masks"
OUT_DIR = DATA_ROOT / "embeddings"


def main() -> None:
    image_ids = sorted(p.stem for p in MASK_DIR.glob("*.png"))
    print(f"{len(image_ids)} images", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "image_ids.npy", np.array(image_ids))

    dataset = ImageIdDataset(IMAGE_DIR, image_ids)
    for name, loader in [("siglip2", load_siglip2), ("tianmu", load_tianmu)]:
        print(f"embedding with {name}", flush=True)
        embed_fn = loader()
        emb = extract_embeddings(embed_fn, dataset)
        np.save(OUT_DIR / f"{name}.npy", emb.numpy())
        print(f"  -> {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
