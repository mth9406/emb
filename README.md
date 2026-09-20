# Silhouette-aware dress image retrieval

This submission fine-tunes the vision encoder of `google/siglip2-so400m-patch14-384` for dress silhouette retrieval. It includes data preparation, positive-pair mining, training, evaluation, catalog indexing, and one-image Top-10 search. `report.md` is intentionally not part of this operational guide.

## Final model and result

- Final checkpoint: [`epoch=0-step=600.ckpt`](https://drive.google.com/drive/folders/1WgiN0qJQCdhQL5YaF2frfEBDlBlrpQuG)
- Base model: `google/siglip2-so400m-patch14-384`
- Fine-tuning: LoRA on the vision attention Q/V projections plus a DINO-style 256-dimensional prototype head.
- Native inference input: dress-only global crop (Grounding-DINO detection, SAM segmentation, background zero-filled).
- Train-corpus silhouette Recall@20: base SigLIP2 raw `0.1344`, Tianmu-MERE raw `0.1155`, fine-tuned SigLIP2 masked `0.1989` (random `0.0050`).

The Drive folder must remain shared as **Anyone with the link / Viewer**; the checkpoint file inside it must be the `step=600` one (swap it if it still holds an older upload). The checkpoint was selected from `experiments/run_1h/checkpoints/epoch=0-step=600.ckpt`, **not the final step=2400 checkpoint the run produced** — Recall@20 peaks at step 600 and falls as training continues past it (prototype usage collapses onto a handful of the 1,024 prototypes; see `report.md` section 3 for the full diagnosis). It was evaluated with the committed `eval.py` settings.

## Environment and assets

The reference runtime is one NVIDIA L40S GPU with the RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0 + CUDA 12.8). CPU-only execution is not supported by the default commands because segmentation and embedding use `--device cuda`.

```bash
cd /workspace/emb/repo
pip install -r requirements.txt

# One-time prerequisite for the public GLAMI image downloader.
sudo apt-get update && sudo apt-get install -y aria2

# Downloads public checkpoints and the GLAMI dresses subset under /workspace/emb.
bash scripts/download_assets.sh
```

Expected  layout:

```text
/workspace/emb/
  repo/          # this repository
  ckpts/         # SigLIP2, Tianmu-MERE, Grounding-DINO, SAM
  data/          # public training images, masks, descriptors, pairs, embeddings
  experiments/   # training checkpoints and evaluation outputs
```

`scripts/download_assets.sh` obtains the public base models and public GLAMI assets; download the final `.ckpt` from the Drive link above into `experiments/run_1h/checkpoints/` (or pass its own path to commands below).

## Reproduce preprocessing, training, and evaluation

The training source is GLAMI-1M `category_name=dresses`. We use a deterministic random sample of 4,000 candidate images (`--limit 4000 --seed 0`); one segmentation failure leaves 3,999 images. The provided 100-image external test set is in `data/raw/images-800px-test/`, physically separated from training and never used for pair mining or fitting.

```bash
# 1. Create binary dress masks. Re-running skips masks already written.
bash scripts/extract_masks.sh train --limit 4000 --seed 0
bash scripts/extract_masks.sh test

# 2. Create 64x64 letterboxed canonical masks; rank pairwise mask IoU;
#    exclude exact name matches; retain IoU in [0.90, 0.95).
bash scripts/mine_pairs.sh

# 3. Optional pipeline smoke test, then the final one-hour L40S run.
bash scripts/train.sh configs/self_distill_v1.yaml
bash scripts/train.sh configs/self_distill_v2.yaml

# 4. Compare base SigLIP2, fine-tuned SigLIP2 (step 600), and Tianmu-MERE.
bash scripts/eval.sh experiments/run_1h/checkpoints/epoch=0-step=600.ckpt
```

The training configuration is committed in `configs/self_distill_v2.yaml`: batch size 8, bf16 mixed precision, 3,000 maximum steps, LoRA rank 16 / alpha 32 / dropout 0.1, flat learning rate `1e-4`, 1,024 prototypes, and teacher EMA momentum `0.996 → 1.0`. The actual run configuration is copied to `experiments/run_1h/config.yaml`.

Similarity is deliberately independent of evaluated model embeddings: it is IoU between each dress mask after crop, square letterboxing, and 64×64 resize. Exact product-name matches and near duplicates above IoU 0.95 are excluded. The train corpus supplies proxy positives for Recall@20; the external 100 images are qualitative cross-domain queries only.

## Build a private catalog and search one image

Create a UTF-8 CSV whose `product_id` values are unique. Image paths may be absolute or relative to the CSV.

```csv
product_id,image_path
SKU-001,images/SKU-001.jpg
SKU-002,images/SKU-002.jpg
```

Build a catalog once. This automatically extracts a dress mask for every catalog image, applies the same masked global-crop preprocessing as fine-tuning, and writes `product_ids.npy`, normalized `embeddings.npy`, masks, failures, and metadata under the output directory.

```bash
bash scripts/build_catalog.sh \
  --catalog /path/to/catalog.csv \
  --checkpoint /path/to/epoch=0-step=600.ckpt \
  --output-dir /path/to/catalog_index
```

Search an arbitrary query image. It automatically detects and segments the dress, embeds the masked crop, computes cosine similarity against the saved catalog, and prints a JSON list of the 10 highest-scoring `product_id` values in descending similarity order.

```bash
bash scripts/search_catalog.sh \
  --image /path/to/private_query.jpg \
  --index-dir /path/to/catalog_index \
  --checkpoint /path/to/epoch=0-step=600.ckpt \
  --top-k 10
```

If no dress is detected, catalog construction records the product in `failed.csv`; a query exits with an error instead of silently using a different preprocessing path.

## Code map

- `src/dataset/`: Grounded-SAM mask extraction, canonical-mask IoU pair mining, and the six-view training dataset.
- `src/model/`, `src/modules/`, `src/func/`: LoRA SigLIP2 student/EMA teacher and DINO loss.
- `train.py`, `eval.py`: experiment execution and the three-model comparison, including qualitative retrieval sheets.
- `src/tools/build_catalog.py`, `src/tools/search_catalog.py`: submission-facing catalog indexing and Top-10 JSON retrieval.
- `configs/`: smoke and final training settings; `scripts/` provides shell entry points for every stage.
