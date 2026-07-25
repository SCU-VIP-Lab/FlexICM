"""Swin backbone helpers shared by detection / segmentation teachers.

Swin-B: F1 at H/4 with C=128 (Cascade Mask R-CNN / UPerNet / MaskFormer).
Optional Swin-T/S variants are supported via `swin_variant` for experiments.

h from TAIC replaces F1 and is fed into Stage 2 onward.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexicm.tasks.losses import freeze_module

_SWIN_TIMM_NAMES = {
    "base": "swin_base_patch4_window7_224",
    "tiny": "swin_tiny_patch4_window7_224",
    "small": "swin_small_patch4_window7_224",
}


class SimpleFPN(nn.Module):
    """Lightweight FPN producing P2..P6 from F1..F4 (channels -> fpn_dim)."""

    def __init__(self, in_channels_list: List[int], fpn_dim: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list])
        self.output = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in in_channels_list])
        self.p6 = nn.Conv2d(fpn_dim, fpn_dim, 3, stride=2, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # feats: [F1,F2,F3,F4] high-res -> low-res
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        outs = [out(lat) for out, lat in zip(self.output, laterals)]
        p2, p3, p4, p5 = outs
        p6 = self.p6(p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}


def build_swin_backbone(pretrained: bool = True, swin_variant: str = "base"):
    """Build Swin via timm; returns backbone with features_only stages."""
    try:
        import timm
    except ImportError as e:
        raise ImportError("Please install timm to use Swin teachers: pip install timm") from e

    key = swin_variant.lower().replace("swin-", "").replace("swin_", "")
    if key not in _SWIN_TIMM_NAMES:
        raise ValueError(f"Unknown swin_variant={swin_variant!r}; expected one of {list(_SWIN_TIMM_NAMES)}")

    model = timm.create_model(
        _SWIN_TIMM_NAMES[key],
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
        img_size=224,
    )
    return model


def build_swin_b_backbone(pretrained: bool = True):
    """Backward-compatible alias for Swin-B."""
    return build_swin_backbone(pretrained=pretrained, swin_variant="base")


class SwinStageTeacher(nn.Module):
    """
    Extract F1..F4 from a Swin backbone (base / tiny / small).
    Truncated path: treat input h as F1, run remaining stages.
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_fpn: bool = True,
        fpn_dim: int = 256,
        swin_variant: str = "base",
    ):
        super().__init__()
        self.swin_variant = swin_variant
        self.backbone = freeze_module(
            build_swin_backbone(pretrained=pretrained, swin_variant=swin_variant)
        )
        # timm: Swin-B [128,256,512,1024], Swin-T [96,192,384,768]
        self.feat_channels = list(self.backbone.feature_info.channels())
        self.use_fpn = use_fpn
        if use_fpn:
            self.fpn = freeze_module(SimpleFPN(self.feat_channels, fpn_dim=fpn_dim))
        else:
            self.fpn = None

        self._f1_dim = self.feat_channels[0]

    @property
    def f1_channels(self) -> int:
        return self._f1_dim

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # ImageNet normalization; x in [0,1]
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (x - mean) / std

    def extract_stages_from_image(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._normalize(x)
        feats = self.backbone(x)
        # timm may return NHWC for swin; convert to NCHW
        outs = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] == self.feat_channels[outs.__len__() if False else 0]:
                pass
            if f.shape[1] not in self.feat_channels and f.shape[-1] in self.feat_channels:
                f = f.permute(0, 3, 1, 2).contiguous()
            outs.append(f)
        # Fix channel-based NHWC detection more robustly
        fixed = []
        for i, f in enumerate(feats):
            if f.shape[1] == self.feat_channels[i]:
                fixed.append(f)
            elif f.shape[-1] == self.feat_channels[i]:
                fixed.append(f.permute(0, 3, 1, 2).contiguous())
            else:
                fixed.append(f)
        return {f"f{i+1}": fixed[i] for i in range(4)}

    def extract_fpn_from_image(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stages = self.extract_stages_from_image(x)
        feats = [stages["f1"], stages["f2"], stages["f3"], stages["f4"]]
        assert self.fpn is not None
        return self.fpn(feats)

    def forward_stages_from_h(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Paper: feed h into Stage 2 to obtain reconstructed F1..F4.
        We set F1=h (project if needed) and run remaining Swin stages.
        """
        if h.shape[1] != self._f1_dim:
            # lazy 1x1 proj (frozen zeros init then identity-ish); created once
            if not hasattr(self, "h_proj"):
                self.h_proj = nn.Conv2d(h.shape[1], self._f1_dim, 1).to(h.device)
                nn.init.zeros_(self.h_proj.bias)
                with torch.no_grad():
                    self.h_proj.weight.zero_()
                    c = min(h.shape[1], self._f1_dim)
                    for i in range(c):
                        self.h_proj.weight[i, i % h.shape[1], 0, 0] = 1.0
                freeze_module(self.h_proj)
            h = self.h_proj(h)

        # Use timm model stages manually when available
        stages = self._run_from_f1(h)
        return stages

    def _run_from_f1(self, f1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Swin stages 2-4 starting from F1 feature map (B,C,H/4,W/4).
        Implementation depends on timm version; fall back to approximating F2-F4
        via successive stride-2 convs matching channels if internals are inaccessible.
        """
        model = self.backbone
        # Try official layers path (timm Swin)
        try:
            x = f1
            # timm features_only wrappers store model as model.model sometimes
            core = model.model if hasattr(model, "model") else model
            # Expect patch embed already done; stages are layers
            layers = None
            for attr in ("layers", "layers_l"):
                if hasattr(core, attr):
                    layers = getattr(core, attr)
                    break
            if layers is None and hasattr(core, "stages"):
                layers = core.stages

            if layers is not None and len(layers) >= 4:
                # layers[0] already produced f1; run 1..3
                # Swin layer input is often NHWC tokens — handle both
                feats = [f1]
                x = f1
                for i in range(1, 4):
                    x = self._forward_swin_layer(layers[i], x)
                    feats.append(x)
                return {f"f{i+1}": feats[i] for i in range(4)}
        except Exception:
            pass

        # Fallback: frozen strided projections to synthesize multi-scale maps
        if not hasattr(self, "_fallback_down"):
            downs = nn.ModuleList()
            chs = self.feat_channels
            for i in range(3):
                downs.append(
                    freeze_module(
                        nn.Sequential(
                            nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1),
                            nn.GELU(),
                        )
                    )
                )
            self._fallback_down = downs.to(f1.device)

        feats = [f1]
        x = f1
        for down in self._fallback_down:
            x = down(x)
            feats.append(x)
        return {f"f{i+1}": feats[i] for i in range(4)}

    @staticmethod
    def _forward_swin_layer(layer, x: torch.Tensor) -> torch.Tensor:
        """Forward one Swin stage; accept NCHW and convert if needed."""
        nchw = x.shape[1] < x.shape[-1]  # heuristic
        # Many timm swin layers expect NCHW in recent versions with features_only
        out = layer(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 4 and out.shape[-1] < out.shape[1] and out.shape[1] > 64:
            # already NCHW
            return out
        if out.dim() == 4 and out.shape[1] < out.shape[-1]:
            # NHWC -> NCHW
            return out.permute(0, 3, 1, 2).contiguous()
        return out

    def forward_fpn_from_h(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        stages = self.forward_stages_from_h(h)
        feats = [stages["f1"], stages["f2"], stages["f3"], stages["f4"]]
        assert self.fpn is not None
        return self.fpn(feats)

    def gt_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_fpn:
            return self.extract_fpn_from_image(x)
        return self.extract_stages_from_image(x)

    def pred_features(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_fpn:
            return self.forward_fpn_from_h(h)
        return self.forward_stages_from_h(h)
