#!/usr/bin/env python3
"""Codec test for TAIC (base layer).

Reports likelihood bpp, feature-alignment distortion D, and optional actual bitstream bpp.
Does NOT compute task metrics (mAP / mIoU / PQ / OKS) — those come later.

Example:
  python scripts/eval_taic.py -c configs/eval/taic_detection.yaml
  python scripts/eval_taic.py -c configs/eval/taic_detection.yaml --actual-bpp
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

from flexicm.data import (
    COCOImageDataset,
    COCOWholeBodyImageDataset,
    ImageFolderDataset,
    build_test_transform,
)
from flexicm.models import TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.losses import TAICCriterion
from flexicm.utils.codec_test import resolve_ckpt, test_taic_loader
from flexicm.utils.train_utils import load_checkpoint_dict, load_yaml_config, set_seed


def parse_args(argv):
    parser = argparse.ArgumentParser("Codec test: FlexICM TAIC")
    parser.add_argument("-c", "--config", required=True, help="configs/eval/taic_*.yaml")
    given, remaining = parser.parse_known_args(argv)
    cfg_path = given.config if os.path.isabs(given.config) else os.path.join(REPO_ROOT, given.config)
    cfg = load_yaml_config(cfg_path)
    parser.set_defaults(**cfg)
    parser.add_argument("--actual-bpp", action="store_true", help="Also run compress/decompress bpp")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for a smoke test")
    parser.add_argument("--split", type=str, default=None, help="Override image split folder, e.g. val2017")
    args = parser.parse_args(remaining)
    args.config = cfg_path
    if given.__dict__.get("actual_bpp") or "--actual-bpp" in argv:
        args.actual_bpp = True
    return args


def build_loader(args, device):
    split = args.split or getattr(args, "split", None) or "val2017"
    tf = build_test_transform()
    root = args.dataset_path
    split_dir = os.path.join(root, split)
    if os.path.isdir(split_dir):
        if args.task == "pose":
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

    task = args.task
    meta = TASK_META[task]
    out_channels = getattr(args, "out_channels", meta["out_channels"])
    align_mode = getattr(args, "align_mode", meta["align_mode"])
    lmbda = getattr(args, "lmbda", 0.0035)

    ckpt = resolve_ckpt(args.checkpoint, REPO_ROOT, label="TAIC checkpoint")
    print(f"Loading TAIC checkpoint: {ckpt}")

    net = TAIC(N=128, M=192, out_channels=out_channels).to(device)
    state, _ = load_checkpoint_dict(ckpt, map_location=device)
    missing = net.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    net.eval()

    teacher = build_teacher(task, pretrained_backbone=getattr(args, "pretrained_backbone", True))
    teacher = teacher.to(device).eval()
    criterion = TAICCriterion(lmbda=lmbda, align_mode=align_mode)

    loader = build_loader(args, device)
    print(f"Test set size: {len(loader.dataset)}  device={device}  task={task}")

    result = test_taic_loader(
        net,
        teacher,
        loader,
        criterion,
        device,
        align_divisor=256,
        run_actual_bpp=bool(getattr(args, "actual_bpp", False)),
        max_batches=args.max_batches,
    )

    print("==== TAIC codec test summary ====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    out_dir = getattr(args, "result_dir", None) or os.path.join(
        REPO_ROOT, "logs", "eval_taic", task, str(getattr(args, "quality_level", 1))
    )
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"codec_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "task": task,
        "checkpoint": ckpt,
        "config": args.config,
        "result": result,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
