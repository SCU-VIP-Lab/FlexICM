#!/usr/bin/env python3
"""ADE20K UPerNet-Swin-B teacher on COCO val: partial-class mIoU (paper-style)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--out",
        default="logs/eval_taic/semantic/baseline_ade_coco_partial_v2.json",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from mmseg.apis import init_model, inference_model
    from flexicm.tasks.metric_runners import SemanticMetricRunner
    from flexicm.data.coco_eval import build_panoptic_semantic_gt_loader

    cfg = (
        "checkpoints/task_networks/semantic/"
        "swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py"
    )
    ckpt = (
        "checkpoints/task_networks/semantic/"
        "upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_"
        "20210526_211650-762e2178.pth"
    )
    root = "/home/zbellay/hdd2/Tianma/dataset/coco2017"
    ann = f"{root}/annotations/panoptic_val2017.json"
    gt_folder = f"{root}/annotations/panoptic_val2017"
    img_dir = f"{root}/val2017"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading model...", flush=True)
    model = init_model(cfg, ckpt, device="cuda:0")
    runner = SemanticMetricRunner(device="cuda:0")
    runner.model = model
    runner.class_names = list(model.dataset_meta.get("classes") or [])
    print("ADE classes", len(runner.class_names), flush=True)

    gt_loader, num_classes = build_panoptic_semantic_gt_loader(ann, gt_folder)
    ignore_label = 255
    lookup, mapped_pairs = runner._build_pred_to_gt_lookup(
        runner.class_names, ann, ignore_label=ignore_label
    )
    assert lookup is not None and mapped_pairs
    eval_classes = sorted({int(x) for x in lookup.tolist() if int(x) != ignore_label})
    eval_mask = np.zeros(num_classes, dtype=bool)
    eval_mask[np.asarray(eval_classes, dtype=np.int64)] = True
    print("mapped_classes", len(mapped_pairs), "eval_classes", len(eval_classes), flush=True)
    print("sample map:", list(mapped_pairs.items())[:8], flush=True)

    with open(ann) as f:
        images = sorted(json.load(f)["images"], key=lambda x: x["id"])
    print("n_images", len(images), flush=True)

    hist = np.zeros((num_classes, num_classes + 1), dtype=np.float64)
    n_err = 0
    n_ok = 0
    t0 = time.time()
    for info in tqdm(images, file=sys.stdout, mininterval=5.0):
        path = f"{img_dir}/{info['file_name']}"
        try:
            result = inference_model(model, path)
            pr = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print("ERR", info["id"], e, flush=True)
            continue

        gt = gt_loader(int(info["id"]))
        valid = (pr >= 0) & (pr < len(lookup))
        remapped = np.full(pr.shape, ignore_label, dtype=np.int64)
        remapped[valid] = lookup[pr[valid]]
        pr = remapped
        if gt.shape != pr.shape:
            continue

        gt_eval = np.asarray(gt, dtype=np.int64)
        in_range = (gt_eval >= 0) & (gt_eval < num_classes)
        mapped_ok = np.zeros(gt_eval.shape, dtype=bool)
        mapped_ok[in_range] = eval_mask[gt_eval[in_range]]
        gt_eval = np.where(mapped_ok, gt_eval, ignore_label)

        mask = (gt_eval != ignore_label) & (gt_eval >= 0) & (gt_eval < num_classes)
        if not np.any(mask):
            n_ok += 1
            continue
        gt_m = gt_eval[mask]
        pr_m = pr[mask].astype(np.int64, copy=False)
        invalid = (pr_m < 0) | (pr_m >= num_classes) | (pr_m == ignore_label)
        pr_m = pr_m.copy()
        pr_m[invalid] = num_classes
        hist += np.bincount(
            gt_m * (num_classes + 1) + pr_m,
            minlength=num_classes * (num_classes + 1),
        ).reshape(num_classes, num_classes + 1)
        n_ok += 1

    elapsed = time.time() - t0
    ious = []
    per_class = {}
    with open(ann) as f:
        categories = json.load(f)["categories"]
    for c in eval_classes:
        tp = float(hist[c, c])
        union = float(hist[c, :].sum() + hist[:num_classes, c].sum() - tp)
        if union <= 0:
            continue
        iou = tp / union
        ious.append(iou)
        per_class[categories[c]["name"]] = iou
    miou = float(np.mean(ious)) if ious else 0.0

    out = {
        "model": "UPerNet + Swin-B (ADE20K)",
        "eval": "COCO panoptic val2017, partial classes (ADE name→COCO)",
        "config": cfg,
        "checkpoint": ckpt,
        "n_images": len(images),
        "n_preds": n_ok,
        "n_errors": n_err,
        "elapsed_sec": elapsed,
        "mIoU": miou,
        "mapped_classes": len(mapped_pairs),
        "eval_classes": len(eval_classes),
        "eval_classes_with_gt": len(ious),
        "per_class_iou": per_class,
        "mapped_pairs": mapped_pairs,
    }
    print(
        f"mIoU={miou:.4f} mapped={len(mapped_pairs)} with_gt={len(ious)} "
        f"time={elapsed/60:.1f}min",
        flush=True,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
