"""DINO-style prototype head: MLP bottleneck -> weight-normalized cluster logits."""

from __future__ import annotations

import torch
from torch import nn

from src.modules.backbone import HIDDEN_SIZE

BOTTLENECK_DIM = 256
NUM_PROTOTYPES = 1024


class PrototypeHead(nn.Module):
    def __init__(
        self, in_dim: int = HIDDEN_SIZE, bottleneck_dim: int = BOTTLENECK_DIM, k: int = NUM_PROTOTYPES
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, k, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = nn.functional.normalize(self.mlp(x), dim=-1)
        logits = self.last_layer(bottleneck)
        return logits, bottleneck
