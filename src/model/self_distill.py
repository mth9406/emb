"""Student/teacher DINO-style self-distillation over mined silhouette pairs."""

from __future__ import annotations

import copy
import math

import torch

from src.func.obj_func import DINOLoss
from src.model.base import BaseLightningModule
from src.modules.backbone import LoRAVisionBackbone
from src.modules.prototype_head import NUM_PROTOTYPES, PrototypeHead
from src.tools.collapse_metrics import PrototypeUsageTracker, batch_mean_entropy, per_sample_entropy


class SelfDistillModule(BaseLightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        student_temp: float = 0.1,
        teacher_temp_start: float = 0.04,
        teacher_temp_end: float = 0.07,
        teacher_temp_warmup_steps: int = 1000,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__(lr)
        self.student_backbone = LoRAVisionBackbone(lora_r, lora_alpha, lora_dropout)
        self.student_head = PrototypeHead()
        self.teacher_backbone = copy.deepcopy(self.student_backbone).requires_grad_(False)
        self.teacher_head = copy.deepcopy(self.student_head).requires_grad_(False)

        self.dino_loss = DINOLoss(
            out_dim=NUM_PROTOTYPES,
            student_temp=student_temp,
            teacher_temp_start=teacher_temp_start,
            teacher_temp_end=teacher_temp_end,
            teacher_temp_warmup_steps=teacher_temp_warmup_steps,
            center_momentum=center_momentum,
        )
        self.usage_tracker = PrototypeUsageTracker(k=NUM_PROTOTYPES)
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end

    def _forward_views(self, backbone, head, views: torch.Tensor) -> torch.Tensor:
        b, n = views.shape[:2]
        flat = views.reshape(b * n, *views.shape[2:])
        feat = backbone(flat)
        logits, _ = head(feat)
        return logits.reshape(b, n, -1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits = self._forward_views(self.teacher_backbone, self.teacher_head, batch["teacher_views"])
        student_logits = self._forward_views(self.student_backbone, self.student_head, batch["student_views"])

        loss = self.dino_loss(student_logits, teacher_logits, self.global_step)
        self.dino_loss.update_center(teacher_logits)

        with torch.no_grad():
            student_probs = torch.softmax(student_logits / self.dino_loss.student_temp, dim=-1)
            self.usage_tracker.update(student_probs)
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/per_sample_entropy", per_sample_entropy(student_probs).mean())
            self.log("train/batch_entropy", batch_mean_entropy(student_probs))
            self.log("train/dead_prototype_ratio", self.usage_tracker.dead_ratio())
        return loss

    def _teacher_momentum(self) -> float:
        max_steps = max(self.trainer.max_steps, 1)
        progress = min(self.global_step / max_steps, 1.0)
        cosine = 0.5 * (1 - math.cos(math.pi * progress))
        return self.momentum_start + cosine * (self.momentum_end - self.momentum_start)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        momentum = self._teacher_momentum()
        with torch.no_grad():
            for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
            for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        self.log("train/teacher_momentum", momentum)

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized bottleneck embedding from the current student -- used for retrieval."""
        feat = self.student_backbone(pixel_values)
        _, bottleneck = self.student_head(feat)
        return bottleneck
