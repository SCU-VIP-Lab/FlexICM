#!/usr/bin/env python3
"""Train FlexICM base-layer TAIC for one of the five machine tasks.

Example:
  python scripts/train_taic.py -c configs/taic/detection.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import yaml

# allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _pre_set_cuda_visible_devices(argv):
    """Must run before importing torch, otherwise gpu_id is ignored."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", required=True)
    given, _ = parser.parse_known_args(argv)
    cfg_path = given.config if os.path.isabs(given.config) else os.path.join(REPO_ROOT, given.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    gpu_id = cfg.get("gpu_id", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return cfg_path, gpu_id


_CFG_PATH, _PHYSICAL_GPU_ID = _pre_set_cuda_visible_devices(sys.argv[1:])

import torch
from torch.utils.data import DataLoader

from flexicm.data import (
    COCOImageDataset,
    COCOWholeBodyImageDataset,
    build_task_aligned_transform,
    collate_expand_pad,
    task_align_geom,
)
from flexicm.models import TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.utils import simulate_task_metric
from flexicm.tasks.losses import TAICCriterion
from flexicm.utils.alignment import Alignment
from flexicm.utils.train_utils import (
    AverageMeter,
    CustomDataParallel,
    adamw_trainable,
    exp_dir,
    load_checkpoint_dict,
    load_yaml_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def parse_args(argv):
    parser = argparse.ArgumentParser("Train FlexICM TAIC")
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    given, remaining = parser.parse_known_args(argv)
    cfg = load_yaml_config(given.config)
    # -c already consumed by the first parse; keep it as a default for the second pass
    parser.set_defaults(config=given.config, **cfg)
    for action in parser._actions:
        if "--config" in action.option_strings:
            action.required = False
            break
    parser.add_argument("-T", "--TEST", action="store_true")
    args = parser.parse_args(remaining)
    args.config = given.config
    return args


@torch.no_grad()
def validate(model, teacher, loader, criterion, device, align_divisor=256):
    model.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for images in loader:
        images = images.to(device)
        align = Alignment(divisor=align_divisor, mode="pad", padding_mode="constant").to(device)
        x = align.align(images)
        out = model(x)
        # Keep teacher + codec features on the aligned (padded) grid so the Swin
        # backbone always sees a patch/window-divisible size. bpp still uses the
        # original pixel count below.
        with torch.no_grad():
            gt = teacher.gt_features(x)
            pred = teacher.pred_features(out["h"])
        N, _, H, W = images.shape
        stats = criterion(out, pred, gt, num_pixels=N * H * W)
        for k in meters:
            meters[k].update(stats[k].item())
    model.train()
    return {k: m.avg for k, m in meters.items()}


def train_one_epoch(model, teacher, loader, optimizer, criterion, device, log_every=50):
    model.train()
    teacher.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for i, images in enumerate(loader):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        with torch.no_grad():
            gt = teacher.gt_features(images)
        pred = teacher.pred_features(out["h"])
        N, _, H, W = images.shape
        stats = criterion(out, pred, gt, num_pixels=N * H * W)
        stats["loss"].backward()
        optimizer.step()
        for k in meters:
            meters[k].update(stats[k].item(), n=images.size(0))
        if i % log_every == 0:
            logging.info(
                f"[{i}/{len(loader)}] loss={meters['loss'].avg:.4f} "
                f"bpp={meters['bpp'].avg:.4f} D={meters['distortion'].avg:.6f}"
            )
    return {k: m.avg for k, m in meters.items()}


def attach_simulated_metric(task, stats, family="taic"):
    stats = dict(stats)
    if "bpp" not in stats:
        return stats
    sim = simulate_task_metric(task, stats["bpp"], family=family)
    if sim is None:
        return stats
    stats["task_metric_sim"] = sim["score"]
    stats["task_metric_name"] = sim["metric"]
    return stats


def main(argv):
    args = parse_args(argv)
    set_seed(getattr(args, "seed", 42))
    out_dir = exp_dir(args.root, args.exp_name, args.quality_level)
    setup_logger(os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S") + ".log"))
    logging.info(f"Config: {args.config}")
    for k, v in sorted(vars(args).items()):
        logging.info(f"{k}: {v}")

    # CUDA_VISIBLE_DEVICES already set before importing torch.
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.info(
            f"Using physical GPU {getattr(args, 'gpu_id', _PHYSICAL_GPU_ID)} "
            f"(visible as cuda:0, name={torch.cuda.get_device_name(0)})"
        )
    else:
        logging.info("Using CPU")

    task = args.task
    meta = TASK_META[task]
    out_channels = getattr(args, "out_channels", meta["out_channels"])
    align_mode = getattr(args, "align_mode", meta["align_mode"])

    # data
    max_long_side = getattr(args, "max_long_side", None)
    # Prefer explicit short_edge / max_long_side; patch_size remains short-edge alias.
    short_edge = getattr(args, "short_edge", None)
    if short_edge is None:
        short_edge = getattr(args, "patch_size", None)
    from flexicm.data.datasets import _MISSING

    max_long_arg = args.max_long_side if hasattr(args, "max_long_side") else _MISSING
    train_tf = build_task_aligned_transform(
        task,
        short_edge=short_edge,
        max_long_side=max_long_arg,
        align_to_task=bool(getattr(args, "align_to_task", True)),
    )
    se, ml = task_align_geom(task, short_edge=short_edge, max_long_side=max_long_arg)
    logging.info(f"Train geom: task={task} short_edge={se} max_long_side={ml}")
    if task == "pose":
        train_set = COCOWholeBodyImageDataset(args.dataset_path, "val2017", train_tf)
    else:
        train_set = COCOImageDataset(args.dataset_path, "val2017", train_tf)
    val_split = getattr(args, "val_split", "val2017")
    val_root = os.path.join(args.dataset_path, val_split)
    if not os.path.isdir(val_root):
        val_root = os.path.join(args.dataset_path, "Kodak") if os.path.isdir(
            os.path.join(args.dataset_path, "Kodak")
        ) else os.path.join(args.dataset_path, "train2017")
    val_tf = build_task_aligned_transform(
        task,
        short_edge=short_edge,
        max_long_side=max_long_arg,
        align_to_task=bool(getattr(args, "align_to_task", True)),
    )
    val_set = COCOImageDataset(
        os.path.dirname(val_root), os.path.basename(val_root), val_tf
    ) if os.path.basename(val_root) in ("train2017", "val2017") else __import__(
        "flexicm.data", fromlist=["ImageFolderDataset"]
    ).ImageFolderDataset(val_root, val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,  # deterministic sample order; no random aug in transform
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_expand_pad,  # expand-resize → pad batch (pose-style)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # models
    net = TAIC(N=128, M=192, out_channels=out_channels).to(device)
    if args.base_codec:
        logging.info(f"Loading base TIC codec from {args.base_codec}")
        state, _ = load_checkpoint_dict(args.base_codec, map_location=device)
        net.load_base_codec(state, strict=False)
    if getattr(args, "freeze_base_codec", True):
        net.freeze_base_codec()
        logging.info("Base TIC codec frozen")
    else:
        for p in net.parameters():
            p.requires_grad = True
        logging.info("Base TIC codec unfrozen; training the entire TAIC model")
    logging.info(
        f"Trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.3f}M / "
        f"total {sum(p.numel() for p in net.parameters())/1e6:.3f}M"
    )

    teacher = build_teacher(
        task,
        pretrained_backbone=getattr(args, "pretrained_backbone", True),
        use_official_teacher=getattr(args, "use_official_teacher", True),
        task_config=getattr(args, "task_config", None),
        task_checkpoint=getattr(args, "task_checkpoint", None),
        device=device,
    )
    teacher = teacher.to(device).eval()

    if args.checkpoint:
        logging.info(f"Resume/load {args.checkpoint}")
        state, raw = load_checkpoint_dict(args.checkpoint, map_location=device)
        net.load_state_dict(state, strict=False)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = adamw_trainable(net, lr=args.learning_rate)
    use_bpp_loss = bool(getattr(args, "use_bpp_loss", True))
    criterion = TAICCriterion(
        lmbda=args.lmbda, align_mode=align_mode, use_bpp_loss=use_bpp_loss
    )
    logging.info(f"use_bpp_loss={use_bpp_loss}  (loss = {'R + λD' if use_bpp_loss else 'λD only'})")

    if args.TEST:
        stats = attach_simulated_metric(task, validate(net, teacher, val_loader, criterion, device))
        logging.info(f"TEST {stats}")
        return

    best = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"===== Epoch {epoch}/{args.epochs} =====")
        train_stats = attach_simulated_metric(
            task, train_one_epoch(net, teacher, train_loader, optimizer, criterion, device)
        )
        if epoch % 20 == 0:
            val_stats = attach_simulated_metric(task, validate(net, teacher, val_loader, criterion, device))
            logging.info(f"train={train_stats} val={val_stats}")
            is_best = val_stats["loss"] < best
            best = min(best, val_stats["loss"])
            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": val_stats["loss"],
                        "args": vars(args),
                    },
                    is_best,
                    out_dir,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
