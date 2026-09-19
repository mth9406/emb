"""Simple collapse indicators for a batch of prototype assignment probabilities."""

from __future__ import annotations

import torch


def per_sample_entropy(probs: torch.Tensor) -> torch.Tensor:
    """probs: (..., K) -> (...) entropy in nats, averaged over any batch/view dims outside."""
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def batch_mean_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of the batch-averaged distribution -- low value means all samples
    agree on the same prototype (a stronger collapse signal than per-sample entropy)."""
    mean_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
    return -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()


class PrototypeUsageTracker:
    """EMA histogram of argmax assignments, accumulated across steps.

    A single step's batch (a few dozen samples) can't cover K=1024 prototypes
    by pigeonhole alone, so "dead" must be judged from usage accumulated over
    many steps, not from one batch's argmax.
    """

    def __init__(self, k: int = 1024, momentum: float = 0.9) -> None:
        self.k = k
        self.momentum = momentum
        self.usage = torch.full((k,), 1.0 / k)

    @torch.no_grad()
    def update(self, probs: torch.Tensor) -> None:
        assignments = probs.reshape(-1, probs.shape[-1]).argmax(dim=-1)
        batch_usage = torch.bincount(assignments, minlength=self.k).float()
        batch_usage = batch_usage / batch_usage.sum()
        self.usage = self.momentum * self.usage + (1 - self.momentum) * batch_usage.to(self.usage.device)

    def dead_ratio(self, threshold: float = 0.5) -> float:
        """threshold is relative to uniform usage (1/k): below threshold/k counts as dead."""
        return (self.usage < threshold / self.k).float().mean().item()
