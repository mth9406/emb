"""Shared optimizer setup for LightningModules that train LoRA + head only."""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class BaseLightningModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-4) -> None:
        super().__init__()
        self.lr = lr

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.trainable_parameters(), lr=self.lr)
