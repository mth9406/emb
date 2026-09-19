"""Train the self-distillation model on mined silhouette pairs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.dataset.view_dataset import MinedPairViewDataset
from src.model.self_distill import SelfDistillModule
from src.tools.fixed_query_callback import FixedQueryCallback

EXPERIMENTS_ROOT = Path("/workspace/emb/experiments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = EXPERIMENTS_ROOT / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    dataset = MinedPairViewDataset(
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        mined_pairs_csv=cfg["data"]["mined_pairs_csv"],
        image_size=cfg["data"]["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
    )

    module = SelfDistillModule(**cfg["model"])

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            every_n_train_steps=cfg["checkpoint"]["every_n_train_steps"],
            save_top_k=-1,
        ),
        FixedQueryCallback(
            image_dir=cfg["data"]["image_dir"],
            mask_dir=cfg["data"]["mask_dir"],
            mined_pairs_csv=cfg["data"]["mined_pairs_csv"],
            out_dir=run_dir / "retrieval_samples",
            image_size=cfg["data"]["image_size"],
            **cfg["fixed_query"],
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=pl.loggers.CSVLogger(save_dir=run_dir, name="logs"),
        callbacks=callbacks,
        **cfg["trainer"],
    )
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
