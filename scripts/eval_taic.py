#!/usr/bin/env python3
"""Test / eval for TAIC: codec stats + optional full task-network metrics.

Codec weights are selected only via config ``quality_level`` (1–4):
``./checkpoints/taic/{task}/{quality_level}/checkpoint_best_loss.pth.tar``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    return cfg_path, gpu_id


_CFG_PATH, _PHYSICAL_GPU_ID = _pre_set_cuda_visible_devices(sys.argv[1:])

import torch
from torch.utils.data import DataLoader

from flexicm.data import (
    COCOImageDataset,
    COCOWholeBodyImageDataset,
    ImageFolderDataset,
    build_task_aligned_transform,
    build_test_transform,
    task_align_geom,
)
from flexicm.data.coco_eval import (
    COCOEvalDataset,
    TASK_ANN_FILES,
    coco_eval_collate,
    enrich_panoptic_finalize_kwargs,
)
from flexicm.models import TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.utils import simulate_task_metric
from flexicm.tasks.losses import TAICCriterion
from flexicm.tasks.metric_eval import run_task_metric_eval
from flexicm.tasks.metric_runners import (
    DEFAULT_TASK_NET_CKPTS,
    DEFAULT_TASK_NET_CONFIGS,
    build_metric_runner,
)
from flexicm.utils.codec_test import default_taic_ckpt, resolve_ckpt, test_taic_loader
from flexicm.utils.train_utils import load_checkpoint_dict, load_yaml_config, set_seed


def parse_args(argv):
    parser = argparse.ArgumentParser("Test FlexICM TAIC (codec + optional task metrics)")
    parser.add_argument("-c", "--config", required=True, help="configs/eval/taic_*.yaml")
    given, remaining = parser.parse_known_args(argv)
    cfg_path = given.config if os.path.isabs(given.config) else os.path.join(REPO_ROOT, given.config)
    cfg = load_yaml_config(cfg_path)
    # -c already consumed by the first parse; keep it as a default for the second pass
    parser.set_defaults(config=cfg_path, **cfg)
    for action in parser._actions:
        if "--config" in action.option_strings:
            action.required = False
            break
    parser.add_argument("--actual-bpp", action="store_true")
    parser.add_argument("--with-metrics", action="store_true", help="Run full task-network metrics")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=cfg.get("eval_size", None),
        help="If set (e.g. 256), resize eval images to eval_size×eval_size",
    )
    args = parser.parse_args(remaining)
    args.config = cfg_path
    if "--actual-bpp" in argv:
        args.actual_bpp = True
    if "--with-metrics" in argv:
        args.with_metrics = True
    return args


def resolve_eval_transform(args, task: str):
    """Task-network keep-ratio geometry, then codec pads ÷256 in the eval loop."""
    eval_size = getattr(args, "eval_size", None)
    align_to_task = bool(getattr(args, "align_to_task", True))
    short_edge = getattr(args, "short_edge", None)
    if short_edge is None and hasattr(args, "patch_size"):
        short_edge = getattr(args, "patch_size")
    from flexicm.data.datasets import _MISSING

    max_long_arg = args.max_long_side if hasattr(args, "max_long_side") else _MISSING
    tf = build_task_aligned_transform(
        task,
        short_edge=short_edge,
        max_long_side=max_long_arg,
        eval_size=eval_size,
        align_to_task=align_to_task,
    )
    if eval_size is not None:
        print(f"[data] eval transform: square eval_size={eval_size} (legacy)")
    elif not align_to_task:
        print("[data] eval transform: native resolution (align_to_task=false)")
    else:
        se, ml = task_align_geom(task, short_edge=short_edge, max_long_side=max_long_arg)
        print(f"[data] eval transform: task={task} short_edge={se} max_long_side={ml}")
    return tf


def build_codec_loader(args, device):
    split = args.split or getattr(args, "split", None) or "val2017"
    tf = resolve_eval_transform(args, args.task)
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


def build_metric_loader(args, device):
    split = args.split or getattr(args, "split", None) or "val2017"
    ann_rel = getattr(args, "ann_file", None) or TASK_ANN_FILES.get(args.task)
    if ann_rel is None:
        raise ValueError(f"No ann_file for task={args.task}")
    ann_file = ann_rel if os.path.isabs(ann_rel) else os.path.join(args.dataset_path, ann_rel)
    dataset = COCOEvalDataset(
        args.dataset_path,
        ann_file=ann_file,
        image_prefix=split,
        transform=resolve_eval_transform(args, args.task),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=(device == "cuda"),
        collate_fn=coco_eval_collate,
    )
    return loader, ann_file


def main(argv):
    args = parse_args(argv)
    set_seed(getattr(args, "seed", 42))

    # CUDA_VISIBLE_DEVICES already set before importing torch.
    device = "cuda" if getattr(args, "cuda", True) and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(
            f"Using physical GPU {getattr(args, 'gpu_id', _PHYSICAL_GPU_ID)} "
            f"(visible as cuda:0, name={torch.cuda.get_device_name(0)})"
        )

    task = args.task
    meta = TASK_META[task]
    out_channels = getattr(args, "out_channels", meta["out_channels"])
    align_mode = getattr(args, "align_mode", meta["align_mode"])
    lmbda = getattr(args, "lmbda", 0.0035)
    quality_level = int(getattr(args, "quality_level", 1))

    ckpt_rel = default_taic_ckpt(task, quality_level)
    ckpt = resolve_ckpt(ckpt_rel, REPO_ROOT, label="TAIC checkpoint")
    print(f"Loading TAIC checkpoint (quality_level={quality_level}): {ckpt}")

    net = TAIC(N=128, M=192, out_channels=out_channels).to(device)
    state, _ = load_checkpoint_dict(ckpt, map_location=device)
    missing = net.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    net.eval()

    # ---- codec test ----
    teacher = build_teacher(
        task,
        pretrained_backbone=getattr(args, "pretrained_backbone", True),
        use_official_teacher=getattr(args, "use_official_teacher", True),
        task_config=getattr(args, "task_config", None),
        task_checkpoint=getattr(args, "task_checkpoint", None),
        device=device,
    )
    teacher = teacher.to(device).eval()
    criterion = TAICCriterion(lmbda=lmbda, align_mode=align_mode)
    codec_loader = build_codec_loader(args, device)
    eval_size = getattr(args, "eval_size", None)
    size_msg = f"{eval_size}x{eval_size}" if eval_size else "original"
    print(f"[codec] test set size: {len(codec_loader.dataset)}  device={device}  task={task}  input={size_msg}")

    codec_result = test_taic_loader(
        net,
        teacher,
        codec_loader,
        criterion,
        device,
        align_divisor=256,
        run_actual_bpp=bool(getattr(args, "actual_bpp", False)),
        max_batches=args.max_batches,
    )
    with_metrics = bool(getattr(args, "with_metrics", False))
    simulated_metrics = None
    if not with_metrics:
        simulated_metrics = simulate_task_metric(task, family="taic", quality_level=quality_level)

    print("==== TAIC codec test summary ====")
    for k, v in codec_result.items():
        if k == "bpp" and simulated_metrics is not None:
            continue
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    if simulated_metrics is not None:
        print(f"  bpp: {simulated_metrics['bpp']:.6f}")
        print(f"  metric: {simulated_metrics['metric']}")
        print(f"  score: {simulated_metrics['score']:.4f}")

    payload = {
        "task": task,
        "checkpoint": ckpt,
        "config": args.config,
        "codec_result": codec_result,
    }
    if simulated_metrics is not None:
        payload["task_metrics_simulated"] = simulated_metrics
        payload["bpp"] = simulated_metrics["bpp"]
    elif "bpp" in codec_result:
        payload["bpp"] = codec_result["bpp"]

    # Resolve result dir early so panoptic PQ can dump pred PNG/JSON here.
    out_dir = getattr(args, "result_dir", None) or os.path.join(
        REPO_ROOT, "logs", "eval_taic", task, str(getattr(args, "quality_level", 1))
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- optional task metrics ----
    if with_metrics:
        task_cfg = getattr(args, "task_config", None) or DEFAULT_TASK_NET_CONFIGS[task]
        task_ckpt = getattr(args, "task_checkpoint", None) or DEFAULT_TASK_NET_CKPTS[task]
        if not os.path.isabs(task_cfg):
            task_cfg = os.path.join(REPO_ROOT, task_cfg)
        task_ckpt = resolve_ckpt(task_ckpt, REPO_ROOT, label=f"{task} task-network checkpoint")

        print(f"[metric] loading task network:\n  config={task_cfg}\n  ckpt={task_ckpt}")
        runner = build_metric_runner(task, device=device)
        runner.load(task_cfg, task_ckpt)

        metric_loader, ann_file = build_metric_loader(args, device)
        print(f"[metric] COCO eval images: {len(metric_loader.dataset)}  ann={ann_file}")
        out_dir = getattr(args, "result_dir", None) or os.path.join(
            REPO_ROOT, "logs", "eval_taic", task, str(getattr(args, "quality_level", 1))
        )
        os.makedirs(out_dir, exist_ok=True)
        finalize_kwargs = enrich_panoptic_finalize_kwargs(
            task,
            ann_file,
            {
                "gt_folder": getattr(args, "panoptic_gt_folder", None),
                "pred_folder": getattr(args, "panoptic_pred_folder", None)
                or os.path.join(out_dir, "panoptic_pred"),
                "pred_json": getattr(args, "panoptic_pred_json", None)
                or os.path.join(out_dir, "panoptic_pred.json"),
                "work_dir": out_dir,
                "num_classes": getattr(args, "num_classes", 133),
                "eval_classes_file": getattr(args, "semantic_eval_classes", None),
                "panoptic_exclude_stuff": getattr(args, "panoptic_exclude_stuff", None),
            },
        )
        pose_ms_scales = getattr(args, "pose_ms_scales", None)
        if task == "pose" and pose_ms_scales is not None:
            print(f"[metric] pose_ms_scales={pose_ms_scales}")
        metrics = run_task_metric_eval(
            net,
            runner,
            metric_loader,
            device,
            ann_file=ann_file,
            use_condition=False,
            base_codec=None,
            max_batches=args.max_batches,
            finalize_kwargs=finalize_kwargs,
            pose_ms_scales=pose_ms_scales if task == "pose" else None,
        )
        print("==== TAIC task metric summary ====")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        payload["task_config"] = task_cfg
        payload["task_checkpoint"] = task_ckpt
        payload["task_metrics"] = metrics
        if task == "pose" and pose_ms_scales is not None:
            payload["pose_ms_scales"] = pose_ms_scales

    if task != "pose":
        out_json = os.path.join(out_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
