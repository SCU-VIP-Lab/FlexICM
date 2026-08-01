"""HRNet multi-branch features for pose teacher alignment and metric from-h."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stem_layer1(backbone: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] → ImageNet-norm → stem → layer1 (256-d)."""
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (image - mean) / std
    x = backbone.relu(backbone.norm1(backbone.conv1(x)))
    x = backbone.relu(backbone.norm2(backbone.conv2(x)))
    return backbone.layer1(x)


def _run_stage2_to_stage4_in(
    backbone: nn.Module, x_list_stage2: List[torch.Tensor]
) -> List[torch.Tensor]:
    """stage2 → stage3 → transition3 inputs for stage4."""
    y_list = backbone.stage2(x_list_stage2)
    x_list = []
    for i in range(int(backbone.stage3_cfg["num_branches"])):
        trans = backbone.transition2[i]
        x_list.append(trans(y_list[-1]) if trans is not None else y_list[i])
    y_list = backbone.stage3(x_list)
    x_list = []
    for i in range(int(backbone.stage4_cfg["num_branches"])):
        trans = backbone.transition3[i]
        x_list.append(trans(y_list[-1]) if trans is not None else y_list[i])
    return x_list


def _stage4_multibranch(
    backbone: nn.Module, x_list: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Run stage4 and return real multi-resolution branches F1..F4.

    HigherHRNet sets the last ``HRModule.multiscale_output=False``, so a plain
    ``backbone.stage4(...)`` only returns the fused high-res map. We still run
    every branch's BASIC blocks, keep F2–F4 as those real lower-res streams
    (32/64/128/256 for W32), and fuse F1 with the official high-res fuse so it
    matches ``backbone.forward`` / the AE head.
    """
    y = x_list
    mods = list(backbone.stage4)
    for mod in mods[:-1]:
        y = mod(y)
    last = mods[-1]
    n = int(last.num_branches)
    branches = [last.branches[i](y[i]) for i in range(n)]

    # Official single-scale fuse → identical to backbone output[0]
    f1 = branches[0]
    if last.fuse_layers is not None and len(last.fuse_layers) >= 1:
        for j in range(1, n):
            fuse_j = last.fuse_layers[0][j]
            f1 = f1 + (fuse_j(branches[j]) if fuse_j is not None else branches[j])
        f1 = last.relu(f1)
    return [f1] + branches[1:]


def hrnet_feats_from_image(backbone: nn.Module, image: torch.Tensor) -> List[torch.Tensor]:
    """Full HRNet stem→stage4; return real multi-branch maps (F1..F4).

    ``image`` is RGB in ``[0, 1]`` (same as codec input). ImageNet norm is applied
    inside. Output branches match HRNet-W32 scales: 32/64/128/256 @ H/4..H/32.
    """
    x = _stem_layer1(backbone, image)
    x_list = []
    for i in range(int(backbone.stage2_cfg["num_branches"])):
        trans = backbone.transition1[i]
        x_list.append(trans(x) if trans is not None else x)
    x_list = _run_stage2_to_stage4_in(backbone, x_list)
    return _stage4_multibranch(backbone, x_list)


def hrnet_feats_from_h(
    backbone: nn.Module,
    h: torch.Tensor,
    image: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Run HRNet stage2–4 with codec feature ``h`` as stage2 branch-0.

    ``h`` is BxC0x(H/4)x(W/4) and replaces ``transition1[0](layer1(stem(x)))``.
    Other stage2 branches still need a layer1 tensor; when ``image`` (RGB in
    [0,1], same spatial size as the padded codec input) is provided, the stem
    + layer1 path builds them. Without ``image``, lower branches are synthesized
    by lifting ``h`` toward 256-d via a pinv of ``transition1[0]``.

    Returns real F1..F4 (same layout as ``hrnet_feats_from_image``). ``feats[0]``
    matches the fused high-res map used by the AE head.
    """
    c0 = int(backbone.stage2_cfg["num_channels"][0])
    if h.shape[1] != c0:
        if h.shape[1] > c0:
            h = h[:, :c0]
        else:
            reps = (c0 + h.shape[1] - 1) // h.shape[1]
            h = h.repeat(1, reps, 1, 1)[:, :c0]

    layer1_feat = None
    if image is not None:
        layer1_feat = _stem_layer1(backbone, image)

    x_list: List[torch.Tensor] = [h]
    for i in range(1, int(backbone.stage2_cfg["num_branches"])):
        trans = backbone.transition1[i]
        if layer1_feat is not None:
            x_list.append(trans(layer1_feat) if trans is not None else layer1_feat)
        else:
            conv0 = backbone.transition1[0][0]
            w1 = conv0.weight.data[:, :, 1, 1]  # [C0, 256]
            wp = torch.linalg.pinv(w1).to(device=h.device, dtype=h.dtype)
            h256 = F.conv2d(h, wp.unsqueeze(-1).unsqueeze(-1))
            x_list.append(trans(h256) if trans is not None else h256)

    x_list = _run_stage2_to_stage4_in(backbone, x_list)
    return _stage4_multibranch(backbone, x_list)


def hrnet_branches_to_dict(feats: List[torch.Tensor]) -> dict:
    """Map branch list → ``{f1..fK}`` for ``FeatureAlignLoss(mode='stages')``."""
    return {f"f{i+1}": feats[i] for i in range(len(feats))}
