#!/usr/bin/env python3
"""Mask2Former Swin-L panoptic → semantic mIoU on COCO panoptic val2017."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

INSTANCE_OFFSET = 1000


def panoptic_to_semantic(panoptic_hw: np.ndarray, num_classes: int, ignore_label: int = 255):
    """Decode mmdet panoptic map (id * 1000 + cat) to contiguous semantic labels."""
    sem = (panoptic_hw % INSTANCE_OFFSET).astype(np.int64)
    # void / unlabeled (often == num_classes)
    sem[(sem < 0) | (sem >= num_classes)] = ignore_label
    return sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--out",
        default="logs/eval_taic/semantic/baseline_mask2former_swinl_from_panoptic.json",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    from flexicm.data.coco_eval import build_panoptic_semantic_gt_loader

    register_all_modules()

    cfg = (
        "checkpoints/task_networks/panoptic/"
        "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py"
    )
    ckpt = (
        "checkpoints/task_networks/panoptic/"
        "mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_"
        "20220407_104949-82f8d28d.pth"
    )
    root = "/media/tianma/datasets/coco2017"
    ann = f"{root}/annotations/panoptic_val2017.json"
    gt_folder = f"{root}/annotations/panoptic_val2017"
    img_dir = f"{root}/val2017"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading Mask2Former Swin-L...", flush=True)
    model = init_detector(cfg, ckpt, device="cuda:0")
    # ckpt metainfo may have typos like ' truck'; strip for matching
    class_names = [str(c).strip() for c in (model.dataset_meta.get("classes") or [])]
    num_classes = len(class_names)
    print("classes", num_classes, flush=True)

    with open(ann) as f:
        coco = json.load(f)
    gt_names = [c["name"] for c in coco["categories"]]
    if class_names == gt_names:
        lookup = None
        print("class order matches panoptic GT", flush=True)
    else:
        # build remap pred_idx -> gt_idx by stripped name
        gt_name2idx = {n: i for i, n in enumerate(gt_names)}
        lookup = np.full(num_classes, 255, dtype=np.int64)
        n_map = 0
        for i, n in enumerate(class_names):
            if n in gt_name2idx:
                lookup[i] = gt_name2idx[n]
                n_map += 1
        print(f"class-order mismatch; name-mapped {n_map}/{num_classes}", flush=True)
        if n_map < num_classes:
            missing = [class_names[i] for i in range(num_classes) if lookup[i] == 255]
            print("unmapped pred classes:", missing, flush=True)

    gt_loader, gt_num = build_panoptic_semantic_gt_loader(ann, gt_folder)
    assert gt_num == len(gt_names)
    ignore_label = 255
    images = sorted(coco["images"], key=lambda x: x["id"])
    print("n_images", len(images), flush=True)

    hist = np.zeros((gt_num, gt_num + 1), dtype=np.float64)
    n_err = 0
    n_ok = 0
    t0 = time.time()
    for info in tqdm(images, file=sys.stdout, mininterval=5.0):
        path = f"{img_dir}/{info['file_name']}"
        try:
            result = inference_detector(model, path)
            pan = result.pred_panoptic_seg.sem_seg.squeeze().detach().cpu().numpy()
            pr = panoptic_to_semantic(pan, num_classes=num_classes, ignore_label=ignore_label)
            if lookup is not None:
                valid = (pr >= 0) & (pr < len(lookup))
                remapped = np.full(pr.shape, ignore_label, dtype=np.int64)
                remapped[valid] = lookup[pr[valid]]
                pr = remapped
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print("ERR", info["id"], e, flush=True)
            continue

        gt = gt_loader(int(info["id"]))
        if gt.shape != pr.shape:
            # rare resize mismatch — skip
            continue

        mask = (gt != ignore_label) & (gt >= 0) & (gt < gt_num)
        if not np.any(mask):
            n_ok += 1
            continue
        gt_m = gt[mask].astype(np.int64, copy=False)
        pr_m = pr[mask].astype(np.int64, copy=False)
        invalid = (pr_m < 0) | (pr_m >= gt_num) | (pr_m == ignore_label)
        pr_m = pr_m.copy()
        pr_m[invalid] = gt_num
        hist += np.bincount(
            gt_m * (gt_num + 1) + pr_m,
            minlength=gt_num * (gt_num + 1),
        ).reshape(gt_num, gt_num + 1)
        n_ok += 1

    elapsed = time.time() - t0
    ious = []
    per_class = {}
    for c, name in enumerate(gt_names):
        tp = float(hist[c, c])
        union = float(hist[c, :].sum() + hist[:gt_num, c].sum() - tp)
        if union <= 0:
            continue
        iou = tp / union
        ious.append(iou)
        per_class[name] = iou
    miou = float(np.mean(ious)) if ious else 0.0

    # Also report ADE-overlap subset if available
    ade_json = "logs/eval_taic/semantic/baseline_ade_coco_partial_v2.json"
    subset_miou = None
    subset_n = None
    if os.path.isfile(ade_json):
        mapped = json.load(open(ade_json))["mapped_pairs"]
        subset_names = set(mapped.values())
        subset = [per_class[n] for n in subset_names if n in per_class]
        if subset:
            subset_miou = float(np.mean(subset))
            subset_n = len(subset)

    out = {
        "model": "Mask2Former + Swin-L (COCO-panoptic) → semantic",
        "eval": "COCO panoptic val2017, all categories as semantic mIoU",
        "config": cfg,
        "checkpoint": ckpt,
        "n_images": len(images),
        "n_preds": n_ok,
        "n_errors": n_err,
        "elapsed_sec": elapsed,
        "mIoU": miou,
        "eval_classes_with_gt": len(ious),
        "num_classes": gt_num,
        "subset_ADE_mapped_mIoU": subset_miou,
        "subset_ADE_mapped_n": subset_n,
        "per_class_iou": per_class,
    }
    print(
        f"mIoU={miou:.4f} classes_with_gt={len(ious)}/{gt_num} "
        f"ADE_subset_mIoU={subset_miou} n={subset_n} time={elapsed/60:.1f}min",
        flush=True,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
