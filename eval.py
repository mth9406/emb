"""Evaluate a fine-tuned checkpoint against the SigLIP2/Tianmu-MERE baselines.

Quantitative: shape Recall@20 on the train corpus (identical corpus and
proxy positives to the M5 baseline), for both raw and masked input -- this
separates "masking alone helps" from "SFT on top of masking helps".

Qualitative:
  - fixed queries: top-5/bottom-5 (masked input, the fine-tuned model's
    native mode) for all three models side by side
  - the queries where fine-tuned vs base-SigLIP2 shape-recall improved or
    regressed the most
  - test-set (100 Pexels photos) images used as cross-domain queries into
    the train corpus -- qualitative only, no mined pairs exist for test
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset.view_dataset import load_image, load_mask
from src.dataset.transforms import global_view
from src.paths import DATA_ROOT
from src.tools.query_report import ShapeIoULookup, render_query_report, stack_model_reports
from src.tools.retrieval import (
    ImageIdDataset,
    MaskedImageIdDataset,
    extract_embeddings,
    load_finetuned,
    load_siglip2,
    load_tianmu,
)
from src.tools.shape_descriptors import to_canonical
from src.tools.shape_recall import load_proxy_positives, per_anchor_recall, recall_at_n

IMAGE_DIR = DATA_ROOT / "raw/images-800px"
MASK_DIR = DATA_ROOT / "masks"
TEST_IMAGE_DIR = DATA_ROOT / "raw/images-800px-test"
TEST_MASK_DIR = DATA_ROOT / "masks-test"
DESCRIPTOR_DIR = DATA_ROOT / "descriptors"
EMBED_CACHE_DIR = DATA_ROOT / "embeddings"


def cached(path: Path, build_fn) -> np.ndarray:
    if path.exists():
        return np.load(path)
    emb = build_fn().numpy()
    np.save(path, emb)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--n-highlight", type=int, default=5)
    parser.add_argument("--n-test-queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt_path.parent.parent / "eval"
    for sub in ["fixed_queries", "improved", "regressed", "test_queries"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_ids = sorted(p.stem for p in MASK_DIR.glob("*.png"))
    cached_ids = list(np.load(EMBED_CACHE_DIR / "image_ids.npy"))
    assert cached_ids == image_ids, "corpus changed since M5 embeddings were cached -- rerun embed_corpus.py"
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    raw_ds = ImageIdDataset(IMAGE_DIR, image_ids)
    masked_ds = MaskedImageIdDataset(IMAGE_DIR, MASK_DIR, image_ids)

    print("building/loading embeddings", flush=True)
    embeddings: dict[str, np.ndarray] = {
        "siglip2_raw": np.load(EMBED_CACHE_DIR / "siglip2.npy"),  # cached from M5
        "tianmu_raw": np.load(EMBED_CACHE_DIR / "tianmu.npy"),
    }
    siglip2_embed = load_siglip2()
    embeddings["siglip2_masked"] = cached(
        EMBED_CACHE_DIR / "siglip2_masked.npy", lambda: extract_embeddings(siglip2_embed, masked_ds)
    )
    tianmu_embed = load_tianmu()
    embeddings["tianmu_masked"] = cached(
        EMBED_CACHE_DIR / "tianmu_masked.npy", lambda: extract_embeddings(tianmu_embed, masked_ds)
    )
    finetuned_embed = load_finetuned(ckpt_path)
    embeddings["finetuned_masked"] = extract_embeddings(finetuned_embed, masked_ds).numpy()
    embeddings["finetuned_raw"] = extract_embeddings(finetuned_embed, raw_ds).numpy()

    # --- 1. quantitative: shape Recall@20, raw vs masked, all three models ---
    positives = load_proxy_positives(DATA_ROOT / "pairs/mined_pairs.csv")
    summary = []
    for name, emb in embeddings.items():
        r = recall_at_n(emb, image_ids, positives, n=20)
        summary.append({"model": name, **r})
        print(f"{name:20s} Recall@20={r['mean_recall']:.4f} "
              f"(random={r['random_baseline']:.4f}, n_anchors={r['n_anchors']})")
    with (out_dir / "recall_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mean_recall", "random_baseline", "n_anchors", "n"])
        writer.writeheader()
        writer.writerows(summary)

    iou_lookup = ShapeIoULookup(DESCRIPTOR_DIR)
    native_models = {
        "SigLIP2 (base)": embeddings["siglip2_masked"],
        "Tianmu-MERE": embeddings["tianmu_masked"],
        "SigLIP2 (fine-tuned)": embeddings["finetuned_masked"],
    }

    def train_query_sheets(query_id: str, models: dict[str, np.ndarray]) -> Image.Image:
        idx = id_to_idx[query_id]
        query_img = Image.open(IMAGE_DIR / f"{query_id}.jpg").convert("RGB")
        sheets = {}
        for name, emb in models.items():
            sims = torch.from_numpy(emb) @ torch.from_numpy(emb[idx])
            sheets[name] = render_query_report(
                query_id, query_img, sims, image_ids, IMAGE_DIR, iou_lookup,
                args.top_k, args.bottom_k, exclude_idx=idx,
            )
        return stack_model_reports(sheets)

    # --- 2. qualitative: fixed queries across all three models ---
    anchors = sorted(positives.keys())
    query_ids = random.Random(args.seed).sample(anchors, min(args.n_queries, len(anchors)))
    for query_id in query_ids:
        train_query_sheets(query_id, native_models).save(out_dir / "fixed_queries" / f"query_{query_id}.jpg", quality=85)

    # --- 3. improved / regressed cases: finetuned vs base SigLIP2, both masked ---
    recall_base = per_anchor_recall(embeddings["siglip2_masked"], image_ids, positives, n=20)
    recall_ft = per_anchor_recall(embeddings["finetuned_masked"], image_ids, positives, n=20)
    deltas = sorted(
        ((a, recall_ft[a] - recall_base[a]) for a in recall_base if a in recall_ft),
        key=lambda x: x[1],
    )
    compare_models = {
        "SigLIP2 (base)": native_models["SigLIP2 (base)"],
        "SigLIP2 (fine-tuned)": native_models["SigLIP2 (fine-tuned)"],
    }
    for anchor_id, delta in deltas[-args.n_highlight:]:
        train_query_sheets(anchor_id, compare_models).save(
            out_dir / "improved" / f"query_{anchor_id}_delta{delta:+.3f}.jpg", quality=85
        )
    for anchor_id, delta in deltas[:args.n_highlight]:
        train_query_sheets(anchor_id, compare_models).save(
            out_dir / "regressed" / f"query_{anchor_id}_delta{delta:+.3f}.jpg", quality=85
        )
    print(f"recall delta (fine-tuned - base): best {deltas[-1]}, worst {deltas[0]}")

    # --- 4. test set: cross-domain queries into the train corpus (qualitative only) ---
    test_ids = sorted(p.stem for p in TEST_MASK_DIR.glob("*.png"))
    test_query_ids = random.Random(args.seed).sample(test_ids, min(args.n_test_queries, len(test_ids)))
    corpus_tensors = {name: torch.from_numpy(embeddings[key]) for name, key in [
        ("SigLIP2 (base)", "siglip2_masked"),
        ("Tianmu-MERE", "tianmu_masked"),
        ("SigLIP2 (fine-tuned)", "finetuned_masked"),
    ]}
    model_embed_fns = {
        "SigLIP2 (base)": siglip2_embed,
        "Tianmu-MERE": tianmu_embed,
        "SigLIP2 (fine-tuned)": finetuned_embed,
    }
    for test_id in test_query_ids:
        test_img = load_image(TEST_IMAGE_DIR, test_id)
        test_mask = load_mask(TEST_MASK_DIR, test_id)
        masked_query_img = global_view(test_img, test_mask)
        query_canonical_flat = to_canonical(test_mask).reshape(-1).astype(np.float32)
        iou_fn = lambda result_id, m=query_canonical_flat: iou_lookup.iou_to_mask(result_id, m)

        sheets = {}
        for name, embed_fn in model_embed_fns.items():
            with torch.inference_mode():
                q_vec = embed_fn([masked_query_img])[0].cpu()
            sims = corpus_tensors[name] @ q_vec
            sheets[name] = render_query_report(
                test_id, masked_query_img, sims, image_ids, IMAGE_DIR, None,
                args.top_k, args.bottom_k, iou_fn=iou_fn,
            )
        stack_model_reports(sheets).save(out_dir / "test_queries" / f"query_{test_id}.jpg", quality=85)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
