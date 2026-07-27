#!/usr/bin/env python3
"""Train FlexICM extension-layer C-TAIC for a multi-task scenario.

Scenarios (paper Sec.IV.A):
  s1: detection (base) -> instance (extension)
  s2: semantic (base)  -> panoptic (extension)
  s3: detection (base) -> pose (extension)

Two-stage training (paper Sec.III.B.3):
  stage1: TAIC mode for extension task (SFMA + Task Connector)
  stage2: condition mode (Prompt Generator + Condition Generator), needs base y_b_hat

Example:
  python scripts/train_ctaic.py -c configs/ctaic/s1_det_instance.yaml --stage 1
  python scripts/train_ctaic.py -c configs/ctaic/s1_det_instance.yaml --stage 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flexicm.data import COCOImageDataset, COCOWholeBodyImageDataset, build_train_transform, build_test_transform
from flexicm.models import CTAIC, TAIC
from flexicm.tasks import TASK_META, build_teacher
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

SCENARIOS = {
    "s1": {"base": "detection", "ext": "instance"},
    "s2": {"base": "semantic", "ext": "panoptic"},
    "s3": {"base": "detection", "ext": "pose"},
}


def parse_args(argv):
    parser = argparse.ArgumentParser("Train FlexICM C-TAIC")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    given, remaining = parser.parse_known_args(argv)
    cfg = load_yaml_config(given.config)
    # -c/--stage already consumed by the first parse; keep them for the second pass
    parser.set_defaults(config=given.config, stage=given.stage, **cfg)
    for action in parser._actions:
        if "--config" in action.option_strings:
            action.required = False
            break
    args = parser.parse_args(remaining)
    args.config = given.config
    args.stage = given.stage
    return args


def build_base_codec(args, device):
    """Frozen base-layer TAIC used to provide y_b_hat."""
    base_task = SCENARIOS[args.scenario]["base"]
    out_ch = TASK_META[base_task]["out_channels"]
    base = TAIC(N=128, M=192, out_channels=out_ch).to(device)
    if args.base_taic_checkpoint:
        state, _ = load_checkpoint_dict(args.base_taic_checkpoint, map_location=device)
        base.load_state_dict(state, strict=False)
    elif args.base_codec:
        state, _ = load_checkpoint_dict(args.base_codec, map_location=device)
        base.load_base_codec(state, strict=False)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


@torch.no_grad()
def encode_base_latent(base_model, images):
    out = base_model(images)
    return out["y_hat"]


def train_one_epoch(stage, ext_model, base_model, teacher, loader, optimizer, criterion, device, log_every=50):
    ext_model.train()
    teacher.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for i, images in enumerate(loader):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        if stage == 1:
            out = ext_model(images, y_b_hat=None, use_condition=False)
        else:
            y_b = encode_base_latent(base_model, images)
            out = ext_model(images, y_b_hat=y_b, use_condition=True)
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
                f"[stage{stage} {i}/{len(loader)}] loss={meters['loss'].avg:.4f} "
                f"bpp={meters['bpp'].avg:.4f} D={meters['distortion'].avg:.6f}"
            )
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(stage, ext_model, base_model, teacher, loader, criterion, device, align_divisor=256):
    ext_model.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for images in loader:
        images = images.to(device)
        # Pad to codec/Swin-friendly size (same as train_taic.validate).
        align = Alignment(divisor=align_divisor, mode="pad", padding_mode="constant").to(device)
        x = align.align(images)
        if stage == 1:
            out = ext_model(x, use_condition=False)
        else:
            y_b = encode_base_latent(base_model, x)
            out = ext_model(x, y_b_hat=y_b, use_condition=True)
        gt = teacher.gt_features(x)
        pred = teacher.pred_features(out["h"])
        N, _, H, W = images.shape
        stats = criterion(out, pred, gt, num_pixels=N * H * W)
        for k in meters:
            meters[k].update(stats[k].item())
    ext_model.train()
    return {k: m.avg for k, m in meters.items()}


def main(argv):
    args = parse_args(argv)
    set_seed(getattr(args, "seed", 42))
    stage = args.stage
    out_dir = exp_dir(args.root, f"{args.exp_name}_stage{stage}", args.quality_level)
    setup_logger(os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S") + ".log"))
    logging.info(f"Scenario {args.scenario}: {SCENARIOS[args.scenario]} stage={stage}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    ext_task = SCENARIOS[args.scenario]["ext"]
    meta = TASK_META[ext_task]
    out_channels = getattr(args, "out_channels", meta["out_channels"])
    align_mode = getattr(args, "align_mode", meta["align_mode"])

    train_tf = build_train_transform(args.patch_size)
    if ext_task == "pose":
        train_set = COCOWholeBodyImageDataset(args.dataset_path, "train2017", train_tf)
    else:
        train_set = COCOImageDataset(args.dataset_path, "train2017", train_tf)
    val_set = COCOImageDataset(args.dataset_path, "val2017", build_test_transform())

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    base_model = build_base_codec(args, device) if stage == 2 else None

    net = CTAIC(N=128, M=192, out_channels=out_channels).to(device)
    if args.base_codec:
        state, _ = load_checkpoint_dict(args.base_codec, map_location=device)
        net.load_base_codec(state, strict=False)

    if stage == 1:
        net.freeze_for_stage1()
        if args.taic_init:
            state, _ = load_checkpoint_dict(args.taic_init, map_location=device)
            net.load_taic_checkpoint(state)
    else:
        # stage2: load stage1 C-TAIC / TAIC weights then freeze for generators
        init_ck = args.stage1_checkpoint or args.taic_init
        if not init_ck:
            raise ValueError("stage2 requires --stage1_checkpoint or taic_init in config")
        state, _ = load_checkpoint_dict(init_ck, map_location=device)
        net.load_taic_checkpoint(state)
        net.freeze_for_stage2()

    logging.info(
        f"Trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.3f}M"
    )

    teacher = build_teacher(
        ext_task,
        pretrained_backbone=getattr(args, "pretrained_backbone", True),
        use_official_teacher=getattr(args, "use_official_teacher", True),
        task_config=getattr(args, "task_config", None),
        task_checkpoint=getattr(args, "task_checkpoint", None),
        device=device,
    )
    teacher = teacher.to(device).eval()

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = adamw_trainable(net, lr=args.learning_rate)
    criterion = TAICCriterion(lmbda=args.lmbda, align_mode=align_mode)

    best = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"===== Stage {stage} Epoch {epoch}/{args.epochs} =====")
        train_stats = train_one_epoch(
            stage, net, base_model, teacher, train_loader, optimizer, criterion, device
        )
        val_stats = validate(stage, net, base_model, teacher, val_loader, criterion, device)
        logging.info(f"train={train_stats} val={val_stats}")
        is_best = val_stats["loss"] < best
        best = min(best, val_stats["loss"])
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "stage": stage,
                    "scenario": args.scenario,
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
