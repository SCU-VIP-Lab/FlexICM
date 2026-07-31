"""Shared utilities for frozen task teachers and feature-alignment losses."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RateLoss(nn.Module):
    def forward(self, likelihoods: Dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
        return sum(
            (torch.log(lik).sum() / (-math.log(2) * num_pixels))
            for lik in likelihoods.values()
        )


class FeatureAlignLoss(nn.Module):
    """MSE feature alignment (paper Eqs. 2-3)."""

    def __init__(self, mode: str = "fpn"):
        """
        Args:
            mode: "fpn" averages P2..P6 (Eq.2); "stages" averages F1..F4 (Eq.3)
        """
        super().__init__()
        assert mode in ("fpn", "stages")
        self.mode = mode

    def forward(self, pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "fpn":
            keys = ["p2", "p3", "p4", "p5", "p6"]
        else:
            keys = ["f1", "f2", "f3", "f4"]
        losses = []
        for k in keys:
            if k not in pred or k not in gt:
                continue
            a, b = pred[k], gt[k]
            if a.shape[-2:] != b.shape[-2:]:
                a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
            losses.append(F.mse_loss(a, b))
        if not losses:
            raise KeyError(f"No overlapping feature keys for mode={self.mode}: pred={pred.keys()} gt={gt.keys()}")
        return torch.stack(losses).mean()


class TAICCriterion(nn.Module):
    """L = R + lambda * D  (paper Eq.1).

    Set ``use_bpp_loss=False`` to drop the rate term (ablation): loss = lambda * D only.
    ``bpp`` is still computed and returned for logging.
    """

    def __init__(self, lmbda: float, align_mode: str = "fpn", use_bpp_loss: bool = True):
        super().__init__()
        self.lmbda = lmbda
        self.use_bpp_loss = bool(use_bpp_loss)
        self.rate = RateLoss()
        self.align = FeatureAlignLoss(mode=align_mode)

    def forward(self, codec_out, pred_feats, gt_feats, num_pixels: int):
        bpp = self.rate(codec_out["likelihoods"], num_pixels)
        dist = self.align(pred_feats, gt_feats)
        if self.use_bpp_loss:
            loss = bpp + self.lmbda * dist
        else:
            loss = self.lmbda * dist
        return {
            "loss": loss,
            "bpp": bpp,
            "distortion": dist,
        }


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module
