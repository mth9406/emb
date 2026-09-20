# Silhouette-aware dress image retrieval

This submission fine-tunes the vision encoder of `google/siglip2-so400m-patch14-384` for dress silhouette retrieval. It includes data preparation, positive-pair mining, training, evaluation, catalog indexing, and one-image Top-10 search. `report.md` is intentionally not part of this operational guide.

There are two ways to use this repository:

- **[Quickstart](#quickstart-search-your-own-images)**: index your own product images and search them with the fine-tuned model. Needs about 6.4 GB of public checkpoints plus the 1.8 GB fine-tuned checkpoint. It does **not** need the training dataset.
- **[Full reproduction](#full-reproduction-preprocessing-training-and-evaluation)**: rerun masking, pair mining, training, and the three-model comparison. Additionally downloads the GLAMI-1M dresses subset and Tianmu-MERE.

## Final model and result

- Final checkpoint: [`epoch=0-step=600.ckpt`](https://drive.google.com/drive/folders/1WgiN0qJQCdhQL5YaF2frfEBDlBlrpQuG)
- Base model: `google/siglip2-so400m-patch14-384`
- Fine-tuning: LoRA on the vision attention Q/V projections plus a DINO-style 256-dimensional prototype head.
- Native inference input: dress-only global crop (Grounding-DINO detection, SAM segmentation, background zero-filled).
- Train-corpus silhouette Recall@20: base SigLIP2 raw `0.1344`, Tianmu-MERE raw `0.1155`, fine-tuned SigLIP2 masked `0.1989` (random `0.0050`).

The checkpoint was selected from `experiments/run_1h/checkpoints/epoch=0-step=600.ckpt`, **not the final step=2400 checkpoint the run produced**. Recall@20 peaks at step 600 and falls as training continues past it, because prototype usage collapses onto a handful of the 1,024 prototypes. See `report.md` section 3 for the full diagnosis.

## Where things live

Every path in the code hangs off a single root, `WORKSPACE_ROOT`, which defaults to `/workspace/emb` (the RunPod Network Volume mount this project was developed on). Set the environment variable to put everything somewhere else:

```bash
export WORKSPACE_ROOT=~/emb     # any writable directory
```

Both sections below assume it is set. The layout underneath it is created by the download scripts:

```text
$WORKSPACE_ROOT/
  ckpts/         # SigLIP2, Grounding-DINO, SAM (+ Tianmu-MERE for full reproduction)
  data/          # GLAMI images, masks, descriptors, pairs, embeddings
  experiments/   # training checkpoints and evaluation outputs
```

The repository itself can be cloned anywhere; it does not have to sit inside `$WORKSPACE_ROOT`.

## Quickstart: search your own images

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. The commands default to `--device cuda`, and the reference runtime is one NVIDIA L40S with the RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0 + CUDA 12.8).

### 2. Download the public checkpoints (about 6.4 GB)

```bash
python scripts/download_checkpoints.py --ckpt-root "$WORKSPACE_ROOT/ckpts" --only inference
```

This fetches the three models inference needs and skips Tianmu-MERE, which is only used by the three-model comparison:

| Model | Hugging Face repo | Size | Used for |
|---|---|---|---|
| SigLIP2 | [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384) | 4.3 GB | the frozen backbone the LoRA adapter sits on |
| Grounding-DINO | [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base) | 894 MB | locating the dress |
| SAM | [`facebook/sam-vit-large`](https://huggingface.co/facebook/sam-vit-large) | 1.2 GB | segmenting it |

Re-running the script skips models already present, so it is safe to repeat after an interrupted download.

### 3. Download the fine-tuned checkpoint (1.8 GB)

Download `epoch=0-step=600.ckpt` from [the Drive folder](https://drive.google.com/drive/folders/1WgiN0qJQCdhQL5YaF2frfEBDlBlrpQuG) and place it here:

```text
$WORKSPACE_ROOT/experiments/run_1h/checkpoints/epoch=0-step=600.ckpt
```

```bash
mkdir -p "$WORKSPACE_ROOT/experiments/run_1h/checkpoints"
# then move the downloaded file into that directory
```

Nothing enforces this location. Every command below takes `--checkpoint`, so you can keep the file anywhere and pass its own path instead. The path above is just what the examples use.

### 4. Build a catalog from your images

Create a UTF-8 CSV whose `product_id` values are unique. Image paths may be absolute or relative to the CSV.

```csv
product_id,image_path
SKU-001,images/SKU-001.jpg
SKU-002,images/SKU-002.jpg
```

```bash
bash scripts/build_catalog.sh \
  --catalog /path/to/catalog.csv \
  --checkpoint "$WORKSPACE_ROOT/experiments/run_1h/checkpoints/epoch=0-step=600.ckpt" \
  --output-dir /path/to/catalog_index
```

This detects and segments a dress in every catalog image, applies the same masked global-crop preprocessing used during fine-tuning, and writes `product_ids.npy`, normalized `embeddings.npy`, masks, `failed.csv`, and `metadata.json` under the output directory. Products with no detected dress are recorded in `failed.csv` and skipped.

### 5. Search with one query image

```bash
bash scripts/search_catalog.sh \
  --image /path/to/private_query.jpg \
  --index-dir /path/to/catalog_index \
  --checkpoint "$WORKSPACE_ROOT/experiments/run_1h/checkpoints/epoch=0-step=600.ckpt" \
  --top-k 10
```

It segments the query the same way, embeds the masked crop, and prints the highest-scoring products as JSON in descending similarity order:

```json
[{"product_id": "SKU-004", "score": 0.975637}, {"product_id": "SKU-000", "score": 0.624309}]
```

If no dress is detected in the query, the command exits with an error rather than silently falling back to different preprocessing. Pass `--save-mask /path/to/mask.png` to inspect what was segmented.

One caveat worth knowing when reading the scores: the fine-tuned model's top results sit very close together in cosine similarity, so the ordering within the top 10 carries little information. `report.md` section 4 measures this.

## Full reproduction: preprocessing, training, and evaluation

This path additionally needs the GLAMI-1M dresses subset and Tianmu-MERE.

```bash
# aria2 is required by the GLAMI image downloader.
sudo apt-get update && sudo apt-get install -y aria2

# Downloads all four checkpoints and the GLAMI dresses subset under $WORKSPACE_ROOT.
bash scripts/download_assets.sh
```

The training source is GLAMI-1M `category_name=dresses`. We use a deterministic random sample of 4,000 candidate images (`--limit 4000 --seed 0`); one segmentation failure leaves 3,999 images. The provided 100-image external test set is in `data/raw/images-800px-test/`, physically separated from training and never used for pair mining or fitting.

```bash
# 1. Create binary dress masks. Re-running skips masks already written.
bash scripts/extract_masks.sh train --limit 4000 --seed 0
bash scripts/extract_masks.sh test

# 2. Create 64x64 letterboxed canonical masks, rank pairwise mask IoU,
#    exclude exact name matches, and retain IoU in [0.90, 0.95).
bash scripts/mine_pairs.sh

# 3. Optional pipeline smoke test, then the run the submission is based on.
bash scripts/train.sh configs/self_distill_v1.yaml
bash scripts/train.sh configs/self_distill_v2.yaml

# 4. Compare base SigLIP2, fine-tuned SigLIP2 (step 600), and Tianmu-MERE.
bash scripts/eval.sh "$WORKSPACE_ROOT/experiments/run_1h/checkpoints/epoch=0-step=600.ckpt"
```

The training configuration is committed in `configs/self_distill_v2.yaml`: batch size 8, bf16 mixed precision, 3,000 maximum steps, LoRA rank 16 / alpha 32 / dropout 0.1, flat learning rate `1e-4`, 1,024 prototypes, and teacher EMA momentum `0.996 → 1.0`. Data paths in the configs are relative to `$WORKSPACE_ROOT`. The actual run configuration is copied to `experiments/<run_name>/config.yaml`.

Similarity is deliberately independent of evaluated model embeddings: it is IoU between each dress mask after crop, square letterboxing, and 64×64 resize. Exact product-name matches and near duplicates above IoU 0.95 are excluded. The train corpus supplies proxy positives for Recall@20; the external 100 images are qualitative cross-domain queries only.

## Code map

- `src/paths.py`: the single definition of `WORKSPACE_ROOT` and the data, checkpoint, and experiment roots derived from it.
- `src/dataset/`: Grounded-SAM mask extraction, canonical-mask IoU pair mining, and the six-view training dataset.
- `src/model/`, `src/modules/`, `src/func/`: LoRA SigLIP2 student, EMA teacher, and DINO loss.
- `train.py`, `eval.py`: experiment execution and the three-model comparison, including qualitative retrieval sheets.
- `src/tools/build_catalog.py`, `src/tools/search_catalog.py`: catalog indexing and Top-10 JSON retrieval.
- `configs/`: smoke and final training settings. `scripts/` provides shell entry points for every stage.
