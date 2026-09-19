"""DINO cross-entropy: sharpened+centered teacher (stop-grad) vs student log-softmax."""

from __future__ import annotations

import torch
from torch import nn


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int = 1024,
        student_temp: float = 0.1,
        teacher_temp_start: float = 0.04,
        teacher_temp_end: float = 0.07,
        teacher_temp_warmup_steps: int = 1000,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_end = teacher_temp_end
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def teacher_temp(self, step: int) -> float:
        t = min(step / self.teacher_temp_warmup_steps, 1.0)
        return self.teacher_temp_start + t * (self.teacher_temp_end - self.teacher_temp_start)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, step: int) -> torch.Tensor:
        """student_logits: (B, S, K), teacher_logits: (B, T, K) -> scalar loss."""
        student_log_probs = torch.log_softmax(student_logits / self.student_temp, dim=-1)
        with torch.no_grad():
            teacher_probs = torch.softmax(
                (teacher_logits - self.center) / self.teacher_temp(step), dim=-1
            )

        total, n_terms = 0.0, 0
        for t in range(teacher_logits.shape[1]):
            for s in range(student_logits.shape[1]):
                total = total + torch.sum(
                    -teacher_probs[:, t] * student_log_probs[:, s], dim=-1
                ).mean()
                n_terms += 1
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.reshape(-1, teacher_logits.shape[-1]).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
