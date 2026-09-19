"""SigLIP2 vision encoder, frozen, with a LoRA adapter on Q/V projections."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel

SIGLIP2_PATH = Path("/workspace/emb/ckpts/siglip2-so400m-patch14-384")
HIDDEN_SIZE = 1152


class LoRAVisionBackbone(nn.Module):
    def __init__(self, r: int = 16, alpha: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        vision_model = AutoModel.from_pretrained(SIGLIP2_PATH, dtype=torch.bfloat16).vision_model
        vision_model.requires_grad_(False)
        lora_config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.vision_model = get_peft_model(vision_model, lora_config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values=pixel_values.to(torch.bfloat16)).pooler_output
        return pooled.float()
