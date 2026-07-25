"""Task-specific frozen teachers for FlexICM feature alignment.

Five tasks (paper Sec.III.A / IV.A):
  1. Object detection      - Cascade Mask R-CNN + Swin-B (official Swin det zoo; mAP-bbox)
  2. Instance segmentation - Cascade Mask R-CNN + Swin-B (same zoo; mAP-mask)
  3. Semantic segmentation - UPerNet + Swin-B (FPN P2-P6)
  4. Panoptic segmentation - MaskFormer + Swin-B (stages F1-F4)
  5. Pose estimation       - HigherHRNet (original HRNet backbone)

Optional full MMDet/MMSeg/MMPose models can be attached for end-task evaluation;
training the codec only needs intermediate feature alignment.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexicm.tasks.losses import freeze_module
from flexicm.tasks.swin_teacher import SwinStageTeacher


class DetectionTeacher(nn.Module):
    """Detection / instance teacher for FPN feature alignment.

    Both tasks use Cascade Mask R-CNN + Swin-B (F1 = 128-d).
    """

    align_mode = "fpn"
    out_channels = 128

    def __init__(self, pretrained_backbone: bool = True, task: str = "detection"):
        super().__init__()
        self.task = task
        self.backbone = SwinStageTeacher(
            pretrained=pretrained_backbone,
            use_fpn=True,
            swin_variant="base",
        )
        freeze_module(self)

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.gt_features(images)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.pred_features(h)


class SemanticSegTeacher(nn.Module):
    """UPerNet style: align FPN features (paper Eq.2)."""

    align_mode = "fpn"
    out_channels = 128

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = SwinStageTeacher(pretrained=pretrained_backbone, use_fpn=True)
        freeze_module(self)

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.gt_features(images)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.pred_features(h)


class PanopticSegTeacher(nn.Module):
    """MaskFormer style: align intermediate stages F1..F4 (paper Eq.3)."""

    align_mode = "stages"
    out_channels = 128

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = SwinStageTeacher(pretrained=pretrained_backbone, use_fpn=False)
        freeze_module(self)

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.gt_features(images)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.pred_features(h)


class HigherHRNetTeacher(nn.Module):
    """Pose estimation teacher with original HRNet backbone (not Swin).

    Aligns multi-resolution HRNet features. h (H/4 x W/4 x C) is projected to
    match the HRNet stem/stage-1 width, then remaining stages produce F1..F4-like maps.
    """

    align_mode = "stages"
    out_channels = 32  # HRNet-W32 stem / stage channels (configurable)

    def __init__(self, width: int = 32, pretrained: bool = True):
        super().__init__()
        self.width = width
        self.out_channels = width
        self.stem = freeze_module(self._build_stem(width))
        self.stage_downs = freeze_module(
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                        nn.BatchNorm2d(width * 2),
                        nn.ReLU(inplace=True),
                    ),
                    nn.Sequential(
                        nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
                        nn.BatchNorm2d(width * 4),
                        nn.ReLU(inplace=True),
                    ),
                    nn.Sequential(
                        nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1),
                        nn.BatchNorm2d(width * 8),
                        nn.ReLU(inplace=True),
                    ),
                ]
            )
        )
        self.h_proj = freeze_module(nn.Conv2d(128, width, 1))
        # Optional: load real HigherHRNet via mmpose if available
        self.mmpose_model = None
        if pretrained:
            self._try_load_mmpose()

    @staticmethod
    def _build_stem(width: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def _try_load_mmpose(self):
        try:
            # Placeholder hook: users can replace with mmpose HigherHRNet
            # e.g. init_pose_model(config, checkpoint)
            self.mmpose_model = None
        except Exception:
            self.mmpose_model = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (x - mean) / std

    def _stages_from_f1(self, f1: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = [f1]
        x = f1
        for down in self.stage_downs:
            x = down(x)
            feats.append(x)
        return {f"f{i+1}": feats[i] for i in range(4)}

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._normalize(images)
        f1 = self.stem(x)
        return self._stages_from_f1(f1)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if h.shape[1] != self.width:
            if h.shape[1] != self.h_proj.in_channels:
                # rebuild projection if codec out_channels differs from default 128
                self.h_proj = freeze_module(nn.Conv2d(h.shape[1], self.width, 1).to(h.device))
            h = self.h_proj(h)
        return self._stages_from_f1(h)


def build_teacher(task: str, **kwargs) -> nn.Module:
    task = task.lower()
    pretrained = kwargs.pop("pretrained_backbone", kwargs.pop("pretrained", True))
    if task in ("detection", "object_detection", "det"):
        return DetectionTeacher(task="detection", pretrained_backbone=pretrained, **kwargs)
    if task in ("instance", "instance_seg", "instance_segmentation"):
        return DetectionTeacher(task="instance", pretrained_backbone=pretrained, **kwargs)
    if task in ("semantic", "semantic_seg", "semantic_segmentation"):
        return SemanticSegTeacher(pretrained_backbone=pretrained, **kwargs)
    if task in ("panoptic", "panoptic_seg", "panoptic_segmentation"):
        return PanopticSegTeacher(pretrained_backbone=pretrained, **kwargs)
    if task in ("pose", "pose_estimation"):
        return HigherHRNetTeacher(pretrained=pretrained, **kwargs)
    raise ValueError(f"Unknown task: {task}")


TASK_META = {
    "detection": {
        "align_mode": "fpn",
        "out_channels": 128,
        "metric": "mAP-bbox",
        "dataset": "coco",
    },
    "instance": {
        "align_mode": "fpn",
        "out_channels": 128,  # Cascade Mask R-CNN + Swin-B F1
        "metric": "mAP-mask",
        "dataset": "coco",
    },
    "semantic": {
        "align_mode": "fpn",
        "out_channels": 128,
        "metric": "mIoU",
        "dataset": "coco",
    },
    "panoptic": {
        "align_mode": "stages",
        "out_channels": 128,
        "metric": "PQ",
        "dataset": "coco",
    },
    "pose": {
        "align_mode": "stages",
        "out_channels": 32,
        "metric": "mAP-OKS",
        "dataset": "coco-wholebody",
    },
}
