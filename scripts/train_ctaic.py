#!/usr/bin/env python3
"""Train FlexICM extension-layer C-TAIC for a multi-task scenario.

Scenarios (paper Sec.IV.A):
  s1: detection (base) -> instance (extension)
  s2: semantic (base)  -> panoptic (extension)
  s3: detection (base) -> pose (extension)

Paper training:
  stage1: extension-task TAIC (train via scripts/train_taic.py; loaded as taic_init)
  stage2: C-TAIC condition mode (Prompt/Condition Generator), needs base y_b_hat

Stage-2 trainable switches (yaml):
  train_enhance_layer: false  # also train full enhance CTAIC (not only generators)
  train_base_layer: false     # jointly train base TAIC with enhance

Example (configs default to stage 2):
  python scripts/train_ctaic.py -c configs/ctaic/s1_det_instance.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import yaml

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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
from flexicm.models import CTAIC, TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.losses import TAICCriterion
from flexicm.utils.alignment import Alignment
from flexicm.utils.train_utils import (
    AverageMeter,
    CustomDataParallel,
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
    # default None so yaml `stage:` is honored unless CLI overrides
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None)
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    given, remaining = parser.parse_known_args(argv)
    cfg = load_yaml_config(given.config)
    # -c/--stage already consumed by the first parse; keep them for the second pass
    parser.set_defaults(config=given.config, **cfg)
    for action in parser._actions:
        if "--config" in action.option_strings:
            action.required = False
            break
    args = parser.parse_args(remaining)
    args.config = given.config
    if given.stage is not None:
        args.stage = given.stage
    else:
        args.stage = int(cfg.get("stage", 2))
    return args


def build_base_codec(args, device, train_base_layer: bool = False):
    """Base-layer TAIC used to provide y_b_hat (frozen unless train_base_layer)."""
    base_task = SCENARIOS[args.scenario]["base"]
    out_ch = TASK_META[base_task]["out_channels"]
    base = TAIC(N=128, M=192, out_channels=out_ch).to(device)
    if args.base_taic_checkpoint:
        state, _ = load_checkpoint_dict(args.base_taic_checkpoint, map_location=device)
        base.load_state_dict(state, strict=False)
    elif args.base_codec:
        state, _ = load_checkpoint_dict(args.base_codec, map_location=device)
        base.load_base_codec(state, strict=False)
    for p in base.parameters():
        p.requires_grad = bool(train_base_layer)
    if train_base_layer:
        base.train()
    else:
        base.eval()
    return base


def encode_base_latent(base_model, images, train_base_layer: bool = False):
    if train_base_layer:
        out = base_model(images)
    else:
        with torch.no_grad():
            out = base_model(images)
    return out["y_hat"]


def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_one_epoch(
    stage,
    ext_model,
    base_model,
    teacher,
    loader,
    optimizer,
    criterion,
    device,
    train_base_layer: bool = False,
    log_every=1000,
):
    ext_model.train()
    if base_model is not None:
        if train_base_layer:
            base_model.train()
        else:
            base_model.eval()
    teacher.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for i, images in enumerate(loader):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        if stage == 1:
            out = ext_model(images, y_b_hat=None, use_condition=False)
        else:
            y_b = encode_base_latent(base_model, images, train_base_layer=train_base_layer)
            out = ext_model(images, y_b_hat=y_b, use_condition=True)
        with torch.no_grad():
            gt = teacher.gt_features(images)
        pred = teacher.pred_features(out["h"], images=images)
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
def validate(
    stage,
    ext_model,
    base_model,
    teacher,
    loader,
    criterion,
    device,
    align_divisor=256,
    pad_mode: str = "corner",
    train_base_layer: bool = False,
):
    ext_model.eval()
    if base_model is not None:
        base_model.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion")}
    for images in loader:
        images = images.to(device)
        if pad_mode == "center":
            x = collate_expand_pad(
                [images[i] for i in range(images.size(0))],
                divisor=align_divisor,
            )
        else:
            # Pad to codec/Swin-friendly size (legacy corner pad).
            align = Alignment(divisor=align_divisor, mode="pad", padding_mode="constant").to(device)
            x = align.align(images)
        if stage == 1:
            out = ext_model(x, use_condition=False)
        else:
            y_b = encode_base_latent(base_model, x, train_base_layer=False)
            out = ext_model(x, y_b_hat=y_b, use_condition=True)
        gt = teacher.gt_features(x)
        pred = teacher.pred_features(out["h"], images=x)
        N, _, H, W = images.shape
        stats = criterion(out, pred, gt, num_pixels=N * H * W)
        for k in meters:
            meters[k].update(stats[k].item())
    ext_model.train()
    if base_model is not None and train_base_layer:
        base_model.train()
    return {k: m.avg for k, m in meters.items()}


def main(argv):
    args = parse_args(argv)
    set_seed(getattr(args, "seed", 42))
    stage = args.stage
    out_dir = exp_dir(args.root, f"{args.exp_name}_stage{stage}", args.quality_level)
    setup_logger(os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S") + ".log"))
    logging.info(f"Scenario {args.scenario}: {SCENARIOS[args.scenario]} stage={stage}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.info(
            f"Using physical GPU {getattr(args, 'gpu_id', _PHYSICAL_GPU_ID)} "
            f"(visible as cuda:0, name={torch.cuda.get_device_name(0)})"
        )

    ext_task = SCENARIOS[args.scenario]["ext"]
    meta = TASK_META[ext_task]
    out_channels = getattr(args, "out_channels", meta["out_channels"])
    align_mode = getattr(args, "align_mode", meta["align_mode"])

    short_edge = getattr(args, "short_edge", None)
    if short_edge is None:
        short_edge = getattr(args, "patch_size", None)
    from flexicm.data.datasets import _MISSING

    max_long_arg = args.max_long_side if hasattr(args, "max_long_side") else _MISSING
    align_to_task = bool(getattr(args, "align_to_task", True))
    train_tf = build_task_aligned_transform(
        ext_task,
        short_edge=short_edge,
        max_long_side=max_long_arg,
        align_to_task=align_to_task,
    )
    se, ml = task_align_geom(ext_task, short_edge=short_edge, max_long_side=max_long_arg)
    logging.info(f"Train geom: task={ext_task} short_edge={se} max_long_side={ml}")
    if ext_task == "pose":
        train_set = COCOWholeBodyImageDataset(args.dataset_path, "train2017", train_tf)
    else:
        train_set = COCOImageDataset(args.dataset_path, "train2017", train_tf)
    val_tf = train_tf
    if ext_task == "pose":
        val_set = COCOWholeBodyImageDataset(args.dataset_path, "val2017", val_tf)
        logging.info(
            f"Val geom (pose): short_edge={se} max_long_side={ml} pad=center/÷256 (match train)"
        )
    else:
        val_set = COCOImageDataset(args.dataset_path, "val2017", val_tf)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True,
        collate_fn=collate_expand_pad,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate_expand_pad if ext_task == "pose" else None,
    )
    val_pad_mode = "center" if ext_task == "pose" else "corner"

    train_enhance_layer = bool(getattr(args, "train_enhance_layer", False))
    train_base_layer = bool(getattr(args, "train_base_layer", False))
    if stage != 2 and (train_enhance_layer or train_base_layer):
        logging.warning(
            "train_enhance_layer / train_base_layer only apply to stage 2; ignoring for stage %s",
            stage,
        )
        train_enhance_layer = False
        train_base_layer = False

    base_model = (
        build_base_codec(args, device, train_base_layer=train_base_layer)
        if stage == 2
        else None
    )

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
        # stage2: load extension TAIC (paper stage1), then set trainable mask
        init_ck = getattr(args, "stage1_checkpoint", None) or args.taic_init
        if not init_ck:
            raise ValueError("stage2 requires taic_init (or stage1_checkpoint) in config")
        state, _ = load_checkpoint_dict(init_ck, map_location=device)
        net.load_taic_checkpoint(state)
        net.configure_stage2_trainable(train_enhance_layer=train_enhance_layer)

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

    # Optimizer over enhance (+ optional base) trainable params
    opt_params = [p for p in net.parameters() if p.requires_grad]
    if base_model is not None and train_base_layer:
        opt_params.extend(p for p in base_model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate, weight_decay=0.01)

    use_bpp_loss = bool(getattr(args, "use_bpp_loss", True))
    criterion = TAICCriterion(
        lmbda=args.lmbda, align_mode=align_mode, use_bpp_loss=use_bpp_loss
    )
    logging.info(
        "trainable: enhance_full=%s base=%s | enhance_params=%s base_params=%s | loss=%s",
        train_enhance_layer,
        train_base_layer,
        _count_trainable(net),
        _count_trainable(base_model) if base_model is not None else 0,
        "R + λD" if use_bpp_loss else "λD only",
    )

    best = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"===== Stage {stage} Epoch {epoch}/{args.epochs} =====")
        train_stats = train_one_epoch(
            stage,
            net,
            base_model,
            teacher,
            train_loader,
            optimizer,
            criterion,
            device,
            train_base_layer=train_base_layer,
        )
        val_stats = validate(
            stage,
            net,
            base_model,
            teacher,
            val_loader,
            criterion,
            device,
            pad_mode=val_pad_mode,
            train_base_layer=train_base_layer,
        )
        logging.info(f"train={train_stats} val={val_stats}")
        is_best = val_stats["loss"] < best
        best = min(best, val_stats["loss"])
        if args.save:
            ckpt = {
                "epoch": epoch,
                "stage": stage,
                "scenario": args.scenario,
                "state_dict": net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": val_stats["loss"],
                "args": vars(args),
                "train_enhance_layer": train_enhance_layer,
                "train_base_layer": train_base_layer,
            }
            if base_model is not None and train_base_layer:
                ckpt["base_state_dict"] = base_model.state_dict()
            save_checkpoint(ckpt, is_best, out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
