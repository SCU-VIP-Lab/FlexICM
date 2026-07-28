#!/usr/bin/env python3
"""Test / eval for C-TAIC: codec stats + optional full task-network metrics.

Examples:
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml --with-metrics
  python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml --no-condition --with-metrics
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
from flexicm.data.coco_eval import (
    COCOEvalDataset,
    TASK_ANN_FILES,
    coco_eval_collate,
    enrich_panoptic_finalize_kwargs,
)
from flexicm.models import CTAIC, TAIC
from flexicm.tasks import TASK_META, build_teacher
from flexicm.tasks.utils import simulate_task_metric
from flexicm.tasks.losses import TAICCriterion
from flexicm.tasks.metric_eval import run_task_metric_eval
from flexicm.tasks.metric_runners import (
    DEFAULT_TASK_NET_CKPTS,
    DEFAULT_TASK_NET_CONFIGS,
    build_metric_runner,
)
from flexicm.utils.codec_test import resolve_ckpt, test_ctaic_loader
from flexicm.utils.train_utils import load_checkpoint_dict, load_yaml_config, set_seed

SCENARIOS = {
    "s1": {"base": "detection", "ext": "instance"},
    "s2": {"base": "semantic", "ext": "panoptic"},
    "s3": {"base": "detection", "ext": "pose"},
}


def parse_args(argv):
    parser = argparse.ArgumentParser("Test FlexICM C-TAIC (codec + optional task metrics)")
    parser.add_argument("-c", "--config", required=True)
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
    parser.add_argument("--no-condition", action="store_true")
    parser.add_argument("--with-metrics", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args(remaining)
    args.config = cfg_path
    if "--actual-bpp" in argv:
        args.actual_bpp = True
    if "--no-condition" in argv:
        args.no_condition = True
    if "--with-metrics" in argv:
        args.with_metrics = True
    return args


def build_codec_loader(args, ext_task, device):
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


def build_metric_loader(args, ext_task, device):
    split = args.split or getattr(args, "split", None) or "val2017"
    ann_rel = getattr(args, "ann_file", None) or TASK_ANN_FILES.get(ext_task)
    ann_file = ann_rel if os.path.isabs(ann_rel) else os.path.join(args.dataset_path, ann_rel)
    dataset = COCOEvalDataset(
        args.dataset_path, ann_file=ann_file, image_prefix=split, transform=build_test_transform()
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

    teacher = build_teacher(
        ext_task,
        pretrained_backbone=getattr(args, "pretrained_backbone", True),
        use_official_teacher=getattr(args, "use_official_teacher", True),
        task_config=getattr(args, "task_config", None),
        task_checkpoint=getattr(args, "task_checkpoint", None),
        device=device,
    )
    teacher = teacher.to(device).eval()
    criterion = TAICCriterion(lmbda=lmbda, align_mode=align_mode)
    codec_loader = build_codec_loader(args, ext_task, device)

    codec_result = test_ctaic_loader(
        net,
        base,
        teacher,
        codec_loader,
        criterion,
        device,
        use_condition=use_condition,
        run_actual_bpp=bool(getattr(args, "actual_bpp", False)),
        max_batches=args.max_batches,
    )
    simulated_metrics = None
    if "bpp" in codec_result:
        simulated_metrics = simulate_task_metric(ext_task, codec_result["bpp"], family="ctaic")

    print("==== C-TAIC codec test summary ====")
    for k, v in codec_result.items():
        if k == "bpp":
            continue  # bpp task score comes from utils curves below
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    if simulated_metrics is not None:
        # Report curve-simulated task metric as the bpp-side result (utils.py).
        print(f"  bpp: {simulated_metrics['score']:.4f}")
        print(f"  metric: {simulated_metrics['metric']}")

    payload = {
        "scenario": scenario,
        "base_task": base_task,
        "ext_task": ext_task,
        "checkpoint": ext_ckpt,
        "base_taic_checkpoint": base_ckpt,
        "config": args.config,
        "codec_result": codec_result,
    }
    if simulated_metrics is not None:
        payload["task_metrics_simulated"] = simulated_metrics
        payload["bpp"] = simulated_metrics["score"]

    if getattr(args, "with_metrics", False):
        task_cfg = getattr(args, "task_config", None) or DEFAULT_TASK_NET_CONFIGS[ext_task]
        task_ckpt = getattr(args, "task_checkpoint", None) or DEFAULT_TASK_NET_CKPTS[ext_task]
        if not os.path.isabs(task_cfg):
            task_cfg = os.path.join(REPO_ROOT, task_cfg)
        task_ckpt = resolve_ckpt(task_ckpt, REPO_ROOT, label=f"{ext_task} task-network checkpoint")

        print(f"[metric] loading extension task network:\n  config={task_cfg}\n  ckpt={task_ckpt}")
        runner = build_metric_runner(ext_task, device=device)
        runner.load(task_cfg, task_ckpt)

        metric_loader, ann_file = build_metric_loader(args, ext_task, device)
        print(f"[metric] COCO eval images: {len(metric_loader.dataset)}  ann={ann_file}")
        finalize_kwargs = enrich_panoptic_finalize_kwargs(
            ext_task,
            ann_file,
            {
                "gt_folder": getattr(args, "panoptic_gt_folder", None),
                "pred_folder": getattr(args, "panoptic_pred_folder", None),
                "num_classes": getattr(args, "num_classes", 133),
            },
        )
        metrics = run_task_metric_eval(
            net,
            runner,
            metric_loader,
            device,
            ann_file=ann_file,
            use_condition=use_condition,
            base_codec=base if use_condition else None,
            max_batches=args.max_batches,
            finalize_kwargs=finalize_kwargs,
        )
        print("==== C-TAIC task metric summary ====")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        payload["task_config"] = task_cfg
        payload["task_checkpoint"] = task_ckpt
        payload["task_metrics"] = metrics

    out_dir = getattr(args, "result_dir", None) or os.path.join(
        REPO_ROOT, "logs", "eval_ctaic", scenario, str(getattr(args, "quality_level", 1))
    )
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
