#!/usr/bin/env python3
"""Convert official Swin-Det Cascade Mask R-CNN ckpt -> mmdet 3.x key layout.

Official source:
  https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth

Only the Swin backbone keys need remapping (layers->stages, attn/mlp naming).
neck / rpn_head / roi_head keys already match mmdet 3 CascadeRCNN.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=os.path.join(REPO_ROOT, "checkpoints/task_networks/detection/model.pth"),
    )
    parser.add_argument(
        "--dst",
        default=os.path.join(REPO_ROOT, "checkpoints/task_networks/detection/model_mmdet3.pth"),
    )
    args = parser.parse_args()

    from mmdet.models.backbones.swin import swin_converter

    raw = torch.load(args.src, map_location="cpu")
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    backbone = {}
    rest = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            backbone[k[len("backbone.") :]] = v
        else:
            rest[k] = v

    converted_backbone = swin_converter(backbone)  # adds 'backbone.' prefix
    out_state = {}
    out_state.update(converted_backbone)
    out_state.update(rest)

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    torch.save({"state_dict": out_state, "meta": {"converted_from": args.src}}, args.dst)
    print(f"Wrote {args.dst}")
    print(f"  backbone keys: {sum(1 for k in out_state if k.startswith('backbone.'))}")
    print(f"  other keys:    {sum(1 for k in out_state if not k.startswith('backbone.'))}")
    print(f"  sample backbone: {[k for k in out_state if k.startswith('backbone.')][:5]}")


if __name__ == "__main__":
    main()
