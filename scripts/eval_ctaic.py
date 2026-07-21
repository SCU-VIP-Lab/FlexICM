#!/usr/bin/env python3
"""Codec test for C-TAIC (extension layer).

Reports extension-layer likelihood bpp and feature distortion D (paper: bpp excludes base layer).
Optional actual bitstream bpp via compress/decompress.
Does NOT compute task metrics (mAP / mIoU / PQ / OKS) — those come later.

Example:
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml --no-condition   # TAIC-mode ablation
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml --actual-bpp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flexicm.data import COCOImageDataset, COCOWholeBodyImageDataset, ImageFolderDataset, build_test_transform
from flexicm.models import CTAIC, TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.losses import TAICCriterion
from flexicm.utils.codec_test import resolve_ckpt, test_ctaic_loader
from flexicm.utils.train_utils import load_checkpoint_dict, load_yaml_config, set_seed

SCENARIOS = {
    "s1": {"base": "detection", "ext": "instance"},
    "s2": {"base": "semantic", "ext": "panoptic"},
    "s3": {"base": "detection", "ext": "pose"},
}


def parse_args(argv):
    parser = argparse.ArgumentParser("Codec test: FlexICM C-TAIC")
    parser.add_argument("-c", "--config", required=True, help="configs/eval/ctaic_*.yaml")
    given, remaining = parser.parse_known_args(argv)
    cfg_path = given.config if os.path.isabs(given.config) else os.path.join(REPO_ROOT, given.config)
    cfg = load_yaml_config(cfg_path)
    parser.set_defaults(**cfg)
    parser.add_argument("--actual-bpp", action="store_true")
    parser.add_argument("--no-condition", action="store_true", help="Disable base-layer conditioning (TAIC mode)")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args(remaining)
    args.config = cfg_path
    if "--actual-bpp" in argv:
        args.actual_bpp = True
    if "--no-condition" in argv:
        args.no_condition = True
    return args


def build_loader(args, ext_task, device):
    split = args.split or getattr(args, "split", None) or "val2017"
    tf = build_test_transform()
    root = args.dataset_path
    split_dir = os.path.join(root, split)
    if os.path.isdir(split_dir):
        if ext_task == "pose":
            dataset = COCOWholeBodyImageDataset(root, split, tf)
        else:
            dataset = COCOImageDataset(root, split, tf)
    else:
        dataset = ImageFolderDataset(root, tf)
    return DataLoader(
        dataset,
        batch_size=getattr(args, "test_batch_size", 1),
        shuffle=False,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=(device == "cuda"),
    )


def main(argv):
    args = parse_args(argv)
    set_seed(getattr(args, "seed", 42))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(args, "gpu_id", 0))
    device = "cuda" if getattr(args, "cuda", True) and torch.cuda.is_available() else "cpu"

    scenario = args.scenario
    base_task = SCENARIOS[scenario]["base"]
    ext_task = SCENARIOS[scenario]["ext"]
    base_meta = TASK_META[base_task]
    ext_meta = TASK_META[ext_task]
    out_channels = getattr(args, "out_channels", ext_meta["out_channels"])
    align_mode = getattr(args, "align_mode", ext_meta["align_mode"])
    lmbda = getattr(args, "lmbda", 0.0035)
    use_condition = not bool(getattr(args, "no_condition", False))

    ext_ckpt = resolve_ckpt(args.checkpoint, REPO_ROOT, label="C-TAIC checkpoint")
    base_ckpt = resolve_ckpt(args.base_taic_checkpoint, REPO_ROOT, label="base TAIC checkpoint")

    print(f"Loading base TAIC ({base_task}): {base_ckpt}")
    base = TAIC(N=128, M=192, out_channels=base_meta["out_channels"]).to(device)
    state, _ = load_checkpoint_dict(base_ckpt, map_location=device)
    base.load_state_dict(state, strict=False)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    print(f"Loading C-TAIC extension ({ext_task}): {ext_ckpt}")
    net = CTAIC(N=128, M=192, out_channels=out_channels).to(device)
    state, _ = load_checkpoint_dict(ext_ckpt, map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()

    teacher = build_teacher(ext_task, pretrained_backbone=getattr(args, "pretrained_backbone", True))
    teacher = teacher.to(device).eval()
    criterion = TAICCriterion(lmbda=lmbda, align_mode=align_mode)

    loader = build_loader(args, ext_task, device)
    print(
        f"Test set size: {len(loader.dataset)}  device={device}  "
        f"scenario={scenario}  use_condition={use_condition}"
    )

    result = test_ctaic_loader(
        net,
        base,
        teacher,
        loader,
        criterion,
        device,
        use_condition=use_condition,
        align_divisor=256,
        run_actual_bpp=bool(getattr(args, "actual_bpp", False)),
        max_batches=args.max_batches,
    )

    print("==== C-TAIC codec test summary ====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    out_dir = getattr(args, "result_dir", None) or os.path.join(
        REPO_ROOT, "logs", "eval_ctaic", scenario, str(getattr(args, "quality_level", 1))
    )
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"codec_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "scenario": scenario,
        "base_task": base_task,
        "ext_task": ext_task,
        "checkpoint": ext_ckpt,
        "base_taic_checkpoint": base_ckpt,
        "config": args.config,
        "result": result,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
