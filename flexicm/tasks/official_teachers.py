"""Teachers that load official task-network backbones (+ necks) for feature alignment.

Training Distortion D is computed against these frozen features so that codec
outputs match the same backbone used at metric evaluation time.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexicm.tasks.losses import freeze_module
from flexicm.tasks.metric_runners import swin_feats_from_h


def _resolve_path(path: Optional[str], repo_root: Optional[str] = None) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def _imagenet_norm_rgb01(x: torch.Tensor) -> torch.Tensor:
    """Normalize RGB float images in [0, 1] with ImageNet mean/std (float space)."""
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def _mmdet_preprocess_rgb01(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Apply DetDataPreprocessor-equivalent norm: RGB [0,1] -> model input."""
    pp = getattr(model, "data_preprocessor", None)
    if pp is None:
        return _imagenet_norm_rgb01(x)
    # DetDataPreprocessor stores mean/std for 0-255 inputs
    mean = getattr(pp, "mean", None)
    std = getattr(pp, "std", None)
    if mean is None or std is None:
        return _imagenet_norm_rgb01(x)
    mean_t = mean.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    std_t = std.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    # x in [0,1] RGB; preprocessor mean/std are for 0-255 RGB after bgr_to_rgb
    return (x * 255.0 - mean_t) / std_t


class OfficialSwinTeacher(nn.Module):
    """Frozen Swin teacher from an official MMDet / MMSeg checkpoint.

    align_mode:
      - ``fpn``: P2..P6 via backbone + FPN/neck (detection / instance / semantic)
      - ``stages``: F1..F4 via backbone stages (panoptic)
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        align_mode: str = "fpn",
        framework: str = "mmdet",
        device: str = "cpu",
        backbone_only: bool = False,
    ):
        super().__init__()
        assert align_mode in ("fpn", "stages")
        self.align_mode = align_mode
        self.framework = framework
        self.config_path = _resolve_path(config_path)
        self.checkpoint_path = _resolve_path(checkpoint_path)
        self.backbone_only = backbone_only
        if not self.config_path or not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"task_config not found: {config_path}")
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"task_checkpoint not found: {checkpoint_path}")

        self.model = self._load_model(device)
        freeze_module(self.model)
        self.out_channels = self._infer_f1_channels()
        self._aux_fpn = None

    def _extract_backbone_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k[len("backbone.") :]] = v
            elif k.startswith("module.backbone."):
                backbone_state[k[len("module.backbone.") :]] = v
        return backbone_state

    def _load_model(self, device: str) -> nn.Module:
        if self.framework == "mmdet":
            from mmdet.apis import init_detector

            if self.backbone_only:
                model = init_detector(self.config_path, checkpoint=None, device=device)
                raw = torch.load(self.checkpoint_path, map_location="cpu")
                state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
                backbone_state = self._extract_backbone_state(state)
                if not backbone_state:
                    raise RuntimeError(
                        f"No backbone.* keys found in checkpoint: {self.checkpoint_path}"
                    )
                missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
                logging.info(
                    "Loaded task-network backbone only: "
                    f"matched={len(backbone_state) - len(unexpected)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
                return model

            return init_detector(self.config_path, self.checkpoint_path, device=device)
        if self.framework == "mmseg":
            from mmseg.apis import init_model

            return init_model(self.config_path, self.checkpoint_path, device=device)
        raise ValueError(f"Unknown framework={self.framework}")

    def _infer_f1_channels(self) -> int:
        bb = self.model.backbone
        if hasattr(bb, "num_features"):
            return int(bb.num_features[0])
        if hasattr(bb, "embed_dims"):
            return int(bb.embed_dims)
        return 128

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return _mmdet_preprocess_rgb01(images, self.model)

    def _backbone_stages_from_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return F1..F4 NCHW from Swin backbone (with per-stage norms)."""
        bb = self.model.backbone
        # Prefer official forward when possible (handles abs pos / dropout)
        if hasattr(bb, "forward") and not hasattr(bb, "stages"):
            feats = bb(x)
            return tuple(feats[:4])

        tokens, hw_shape = bb.patch_embed(x)
        if getattr(bb, "use_abs_pos_embed", False):
            tokens = tokens + bb.absolute_pos_embed
        tokens = bb.drop_after_pos(tokens)
        outs = []
        for i, stage in enumerate(bb.stages):
            tokens, hw_shape, out, out_hw = stage(tokens, hw_shape)
            if i in getattr(bb, "out_indices", (0, 1, 2, 3)):
                norm = getattr(bb, f"norm{i}")
                out = norm(out)
                feat = (
                    out.view(-1, *out_hw, bb.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(feat)
        return tuple(outs)

    def _fpn_from_stages(self, stages: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        neck = getattr(self.model, "neck", None)
        if neck is None and self.framework == "mmseg":
            # UPerNet has no separate FPN neck; reuse Cascade Mask R-CNN FPN
            # (same Swin-B channel layout) so Eq.2 aligns P2..P6.
            neck = self._get_or_build_aux_fpn(stages[0].device)
        if neck is None:
            p = list(stages[:4])
            while len(p) < 4:
                p.append(p[-1])
            p2, p3, p4, p5 = p
            p6 = F.avg_pool2d(p5, kernel_size=2, stride=2)
            return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}

        pyramid = neck(stages)
        keys = ["p2", "p3", "p4", "p5", "p6"]
        out = {}
        for i, feat in enumerate(pyramid):
            if i < len(keys):
                out[keys[i]] = feat
        return out

    def _get_or_build_aux_fpn(self, device) -> nn.Module:
        if hasattr(self, "_aux_fpn") and self._aux_fpn is not None:
            return self._aux_fpn
        from mmdet.models.necks import FPN

        fpn = FPN(
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=5,
        )
        # Load official Cascade Mask R-CNN neck weights when available
        det_ckpt = _resolve_path("checkpoints/task_networks/detection/model_mmdet3.pth")
        if det_ckpt and os.path.isfile(det_ckpt):
            raw = torch.load(det_ckpt, map_location="cpu")
            state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            neck_state = {
                k[len("neck.") :]: v for k, v in state.items() if k.startswith("neck.")
            }
            missing, unexpected = fpn.load_state_dict(neck_state, strict=False)
            # missing/unexpected are fine to ignore for BN tracking buffers etc.
        self._aux_fpn = freeze_module(fpn.to(device))
        return self._aux_fpn

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._preprocess(images)
        if (
            self.align_mode == "fpn"
            and hasattr(self.model, "extract_feat")
            and getattr(self.model, "neck", None) is not None
        ):
            # CascadeRCNN.extract_feat = neck(backbone(x)) -> P2..P6
            feats = self.model.extract_feat(x)
            keys = ["p2", "p3", "p4", "p5", "p6"]
            return {keys[i]: feats[i] for i in range(min(len(keys), len(feats)))}

        if self.framework == "mmseg" and hasattr(self.model, "extract_feat"):
            stages = tuple(self.model.extract_feat(x)[:4])
        else:
            stages = self._backbone_stages_from_image(x)
        if self.align_mode == "stages":
            return {f"f{i+1}": stages[i] for i in range(min(4, len(stages)))}
        return self._fpn_from_stages(stages)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        stages = swin_feats_from_h(self.model.backbone, h)
        if self.align_mode == "stages":
            return {f"f{i+1}": stages[i] for i in range(min(4, len(stages)))}
        return self._fpn_from_stages(stages)


class OfficialHRNetTeacher(nn.Module):
    """Frozen HRNet / HigherHRNet-style teacher from MMPose (stages F1..F4)."""

    align_mode = "stages"
    out_channels = 32

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        width: int = 32,
    ):
        super().__init__()
        self.config_path = _resolve_path(config_path)
        self.checkpoint_path = _resolve_path(checkpoint_path)
        if not self.config_path or not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"task_config not found: {config_path}")
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"task_checkpoint not found: {checkpoint_path}")

        from mmpose.apis import init_model

        self.model = init_model(self.config_path, self.checkpoint_path, device=device)
        freeze_module(self.model)
        self.width = width
        self.out_channels = width
        # projection when codec h channels != stem width
        self.h_proj = freeze_module(nn.Conv2d(128, width, 1))

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return _imagenet_norm_rgb01(images)

    def _stages_from_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bb = self.model.backbone if hasattr(self.model, "backbone") else self.model
        # HRNet forward usually returns multi-resolution list / tensor
        feats = bb(x)
        if isinstance(feats, (list, tuple)):
            # take highest-res stream as F1, then downsample proxies for F2..F4
            f1 = feats[0] if feats[0].dim() == 4 else feats[0][-1]
            outs = [f1]
            cur = f1
            for i in range(1, 4):
                if i < len(feats) and isinstance(feats[i], torch.Tensor) and feats[i].dim() == 4:
                    outs.append(feats[i])
                    cur = feats[i]
                else:
                    cur = F.avg_pool2d(cur, 2, 2)
                    outs.append(cur)
            return {f"f{i+1}": outs[i] for i in range(4)}
        # single tensor: synthesize pyramid
        f1 = feats
        outs = [f1]
        cur = f1
        for _ in range(3):
            cur = F.avg_pool2d(cur, 2, 2)
            outs.append(cur)
        return {f"f{i+1}": outs[i] for i in range(4)}

    def gt_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._stages_from_backbone(self._preprocess(images))

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if h.shape[1] != self.width:
            if self.h_proj.in_channels != h.shape[1]:
                self.h_proj = freeze_module(nn.Conv2d(h.shape[1], self.width, 1).to(h.device))
            h = self.h_proj(h)
        # Approximate remaining stages with stride-2 pools (HRNet full from-h
        # needs model-specific stem hooks; pyramid still provides multi-scale D).
        outs = [h]
        cur = h
        for _ in range(3):
            cur = F.avg_pool2d(cur, 2, 2)
            outs.append(cur)
        return {f"f{i+1}": outs[i] for i in range(4)}
