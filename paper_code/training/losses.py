"""
src/losses.py

Loss functions for imbalanced temporal AKI prediction.
All losses mask padded positions (y == -1).

Losses
------
bce_weighted : standard BCE with pos_weight for class imbalance
focal        : focal loss, down-weights easy negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_mask(loss: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
    """Zero out loss at padded positions (targets == -1)."""
    mask = (targets >= 0).float()
    return (loss * mask).sum() / (mask.sum() + 1e-9)


class WeightedBCELoss(nn.Module):
    """
    BCE with logits + pos_weight.
    pos_weight = neg_count / pos_count from training set.
    """
    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        pw   = torch.tensor([self.pos_weight],
                            device=logits.device, dtype=logits.dtype)
        t    = targets.clamp(min=0)   # avoid NaN at padding positions
        loss = F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=pw, reduction='none'
        )
        return _apply_mask(loss, targets)


class FocalLoss(nn.Module):
    """
    Focal loss — down-weights easy negatives so model focuses on
    hard-to-classify examples. Recommended for severe imbalance.

    alpha : weight for positive class (typically pos_rate or 0.25)
    gamma : focusing strength (2.0 is standard; try 1.0–3.0)

    Reference: Lin et al., RetinaNet 2017
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        t    = targets.clamp(min=0)
        bce  = F.binary_cross_entropy_with_logits(
            logits, t, reduction='none'
        )
        prob    = torch.sigmoid(logits)
        pt      = torch.where(t == 1, prob, 1 - prob)
        alpha_t = torch.where(
            t == 1,
            torch.full_like(logits, self.alpha),
            torch.full_like(logits, 1.0 - self.alpha),
        )
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return _apply_mask(loss, targets)


def build_loss(loss_name: str,
               pos_weight: float,
               focal_alpha: float = None,
               focal_gamma: float = 2.0) -> nn.Module:
    """
    Build loss function by name.

    loss_name   : 'bce_weighted' | 'focal'
    pos_weight  : neg/pos ratio from training set
    focal_alpha : if None, computed from pos_weight as pos/(pos+neg)
    focal_gamma : focal loss focusing parameter
    """
    name = loss_name.lower()
    if name == 'bce_weighted':
        print(f"  Loss: WeightedBCE (pos_weight={pos_weight:.1f})")
        return WeightedBCELoss(pos_weight=pos_weight)

    elif name == 'focal':
        if focal_alpha is None:
            # Derive alpha from class ratio
            # pos_weight = neg/pos, so pos_rate = 1/(1+pos_weight)
            focal_alpha = 1.0 / (1.0 + pos_weight)
        print(f"  Loss: FocalLoss "
              f"(alpha={focal_alpha:.3f}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Choose: bce_weighted | focal"
        )
