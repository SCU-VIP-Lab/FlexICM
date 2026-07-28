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
from flexicm.tasks.official_teachers import OfficialHRNetTeacher, OfficialSwinTeacher

# Default official task-network assets (used when use_official_teacher=True)
DEFAULT_TEACHER_ASSETS = {
    "detection": {
        "task_config": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",
        "task_checkpoint": "checkpoints/task_networks/detection/model_mmdet3.pth",
        "framework": "mmdet",
        "align_mode": "fpn",
    },
    "instance": {
        "task_config": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",
        "task_checkpoint": "checkpoints/task_networks/instance/model_mmdet3.pth",
        "framework": "mmdet",
        "align_mode": "fpn",
    },
    "semantic": {
        "task_config": "checkpoints/task_networks/semantic/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py",
        "task_checkpoint": "checkpoints/task_networks/semantic/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth",
        "framework": "mmseg",
        "align_mode": "fpn",
    },
    "panoptic": {
        "task_config": "checkpoints/task_networks/panoptic/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic.py",
        "task_checkpoint": "checkpoints/task_networks/panoptic/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic_20220329_230021-05ec7315.pth",
        "framework": "mmdet",
        "align_mode": "stages",
    },
    "pose": {
        "task_config": "checkpoints/task_networks/pose/ae_hrnet-w32_8xb24-300e_coco-512x512.py",
        "task_checkpoint": "checkpoints/task_networks/pose/hrnet_w32_coco_512x512-bcb8c247_20200816.pth",
        "framework": "mmpose",
        "align_mode": "stages",
    },
}


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
    use_official = kwargs.pop("use_official_teacher", True)
    task_config = kwargs.pop("task_config", None)
    task_checkpoint = kwargs.pop("task_checkpoint", None)
    device = kwargs.pop("device", "cpu")

    # Normalize aliases
    alias = {
        "object_detection": "detection",
        "det": "detection",
        "instance_seg": "instance",
        "instance_segmentation": "instance",
        "semantic_seg": "semantic",
        "semantic_segmentation": "semantic",
        "panoptic_seg": "panoptic",
        "panoptic_segmentation": "panoptic",
        "pose_estimation": "pose",
    }
    task = alias.get(task, task)

    if use_official:
        assets = DEFAULT_TEACHER_ASSETS.get(task)
        if assets is None:
            raise ValueError(f"Unknown task for official teacher: {task}")
        cfg = task_config or assets["task_config"]
        ckpt = task_checkpoint or assets["task_checkpoint"]
        align_mode = kwargs.pop("align_mode", assets["align_mode"])
        framework = assets["framework"]
        if framework in ("mmdet", "mmseg"):
            backbone_only = kwargs.pop("teacher_backbone_only", False)
            teacher = OfficialSwinTeacher(
                config_path=cfg,
                checkpoint_path=ckpt,
                align_mode=align_mode,
                framework=framework,
                device=device,
                backbone_only=bool(backbone_only),
            )
            freeze_module(teacher)
            return teacher
        if framework == "mmpose":
            width = kwargs.pop("width", 32)
            teacher = OfficialHRNetTeacher(
                config_path=cfg,
                checkpoint_path=ckpt,
                device=device,
                width=width,
            )
            freeze_module(teacher)
            return teacher

    # Legacy timm / in-repo teachers (ImageNet Swin / stem HRNet)
    if task == "detection":
        return DetectionTeacher(task="detection", pretrained_backbone=pretrained, **kwargs)
    if task == "instance":
        return DetectionTeacher(task="instance", pretrained_backbone=pretrained, **kwargs)
    if task == "semantic":
        return SemanticSegTeacher(pretrained_backbone=pretrained, **kwargs)
    if task == "panoptic":
        return PanopticSegTeacher(pretrained_backbone=pretrained, **kwargs)
    if task == "pose":
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
