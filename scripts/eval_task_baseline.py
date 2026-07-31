#!/usr/bin/env python3
"""Run official task networks on original COCO images (no codec).

This gives the uncompressed baseline for FlexICM rate–accuracy curves.

Examples:
  python scripts/eval_task_baseline.py --tasks detection,instance
  python scripts/eval_task_baseline.py --tasks detection --max-images 100 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flexicm.utils.codec_test import resolve_ckpt


DEFAULTS = {
    "detection": {
        "metric": "mAP-bbox",
        "ann_file": "annotations/instances_val2017.json",
        "task_config": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",
        "task_checkpoint": "checkpoints/task_networks/detection/model_mmdet3.pth",
        "framework": "mmdet",
    },
    "instance": {
        "metric": "mAP-mask",
        "ann_file": "annotations/instances_val2017.json",
        "task_config": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",
        "task_checkpoint": "checkpoints/task_networks/instance/model_mmdet3.pth",
        "framework": "mmdet",
    },
    "semantic": {
        "metric": "mIoU",
        "ann_file": "annotations/panoptic_val2017.json",
        "task_config": (
            "checkpoints/task_networks/semantic/"
            "swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py"
        ),
        "task_checkpoint": (
            "checkpoints/task_networks/semantic/"
            "upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_"
            "20210526_211650-762e2178.pth"
        ),
        "framework": "mmseg",
        "note": "Current ckpt is ADE20K-pretrained; COCO mIoU needs remapping / COCO semantic GT.",
    },
    "panoptic": {
        "metric": "PQ",
        "ann_file": "annotations/panoptic_val2017.json",
        "task_config": (
            "checkpoints/task_networks/panoptic/"
            "mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic.py"
        ),
        "task_checkpoint": (
            "checkpoints/task_networks/panoptic/"
            "mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic_"
            "20220329_230021-05ec7315.pth"
        ),
        "framework": "mmdet",
        "note": "Needs panoptic_val2017.json + panoptic PNG folder.",
    },
    "pose": {
        "metric": "mAP-OKS",
        "ann_file": "annotations/person_keypoints_val2017.json",
        "task_config": "checkpoints/task_networks/pose/ae_hrnet-w32_8xb24-300e_coco-512x512.py",
        "task_checkpoint": (
            "checkpoints/task_networks/pose/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"
        ),
        "framework": "mmpose",
        "note": "Config/ckpt are AE-HRNet style; may differ from paper HigherHRNet WholeBody.",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        type=str,
        default="detection,instance",
        help="Comma-separated tasks: detection,instance,semantic,panoptic,pose",
    )
    p.add_argument("--dataset-path", type=str, default="/data/Dataset/coco2017")
    p.add_argument("--split", type=str, default="val2017")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--score-thr", type=float, default=0.05)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--wait-gpu-mem-mb",
        type=int,
        default=8000,
        help="If GPU free memory is below this, wait (0 disables waiting).",
    )
    return p.parse_args(argv)


def _gpu_free_mb(gpu_id: int) -> Optional[int]:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(gpu_id: int, need_mb: int):
    if need_mb <= 0:
        return
    while True:
        free = _gpu_free_mb(gpu_id)
        if free is None or free >= need_mb:
            print(f"[gpu] free={free} MiB (>= {need_mb}); starting")
            return
        print(f"[gpu] free={free} MiB < {need_mb}; waiting 60s...")
        time.sleep(60)


def list_coco_images(ann_file: str, max_images: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(ann_file) as f:
        coco = json.load(f)
    images = list(coco["images"])
    images.sort(key=lambda x: int(x["id"]))
    if max_images is not None:
        images = images[:max_images]
    return images


def _inst_to_coco_bbox_mask(
    image_id: int,
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    cat_ids: List[int],
    masks: Optional[Any] = None,
    score_thr: float = 0.05,
) -> tuple[List[Dict], List[Dict]]:
    bbox_results: List[Dict] = []
    mask_results: List[Dict] = []
    for i, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
        if float(score) < score_thr:
            continue
        label = int(label)
        cat_id = int(cat_ids[label]) if label < len(cat_ids) else label + 1
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bbox_results.append(
            {
                "image_id": int(image_id),
                "category_id": cat_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
        if masks is not None:
            try:
                from pycocotools import mask as mask_util

                m = masks[i]
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                m = np.asarray(m).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(m))
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                mask_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": cat_id,
                        "segmentation": rle,
                        "score": float(score),
                    }
                )
            except Exception:
                pass
    return bbox_results, mask_results


def eval_detection_instance(
    task: str,
    dataset_path: str,
    split: str,
    device: str,
    max_images: Optional[int],
    score_thr: float,
) -> Dict[str, Any]:
    from mmdet.apis import inference_detector, init_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg = DEFAULTS[task]
    config = resolve_ckpt(cfg["task_config"], REPO_ROOT, label=f"{task} config")
    ckpt = resolve_ckpt(cfg["task_checkpoint"], REPO_ROOT, label=f"{task} checkpoint")
    ann_file = cfg["ann_file"]
    if not os.path.isabs(ann_file):
        ann_file = os.path.join(dataset_path, ann_file)
    if not os.path.isfile(ann_file):
        return {"error": f"missing ann_file: {ann_file}"}

    model = init_detector(config, ckpt, device=device)
    model.eval()
    images = list_coco_images(ann_file, max_images=max_images)
    coco_gt = COCO(ann_file)
    cat_ids = coco_gt.getCatIds()

    bbox_results: List[Dict] = []
    mask_results: List[Dict] = []
    img_dir = os.path.join(dataset_path, split)

    for info in tqdm(images, desc=f"baseline:{task}"):
        path = os.path.join(img_dir, info["file_name"])
        result = inference_detector(model, path)
        if hasattr(result, "pred_instances"):
            inst = result.pred_instances
            bboxes = inst.bboxes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            labels = inst.labels.detach().cpu().numpy()
            masks = inst.masks if hasattr(inst, "masks") and inst.masks is not None else None
            if masks is not None and hasattr(masks, "to_ndarray"):
                masks = masks.to_ndarray()
            elif masks is not None and hasattr(masks, "detach"):
                masks = masks.detach().cpu().numpy()
        else:
            # unexpected format
            continue
        b, m = _inst_to_coco_bbox_mask(
            info["id"], bboxes, scores, labels, cat_ids, masks=masks, score_thr=score_thr
        )
        bbox_results.extend(b)
        mask_results.extend(m)

    out: Dict[str, Any] = {
        "task": task,
        "n_images": len(images),
        "n_bbox_dets": len(bbox_results),
        "n_mask_dets": len(mask_results),
        "checkpoint": ckpt,
        "ann_file": ann_file,
    }
    if not bbox_results:
        out["error"] = "no detections"
        return out

    if task == "detection":
        dt = coco_gt.loadRes(bbox_results)
        ev = COCOeval(coco_gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out["mAP-bbox"] = float(ev.stats[0])
        out["mAP-bbox@50"] = float(ev.stats[1])
    else:
        if not mask_results:
            out["error"] = "no mask detections"
            return out
        dt = coco_gt.loadRes(mask_results)
        ev = COCOeval(coco_gt, dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out["mAP-mask"] = float(ev.stats[0])
        out["mAP-mask@50"] = float(ev.stats[1])
        # also report bbox for reference
        dt_b = coco_gt.loadRes(bbox_results)
        ev_b = COCOeval(coco_gt, dt_b, iouType="bbox")
        ev_b.evaluate()
        ev_b.accumulate()
        ev_b.summarize()
        out["mAP-bbox"] = float(ev_b.stats[0])
    return out


def eval_pose(
    dataset_path: str,
    split: str,
    device: str,
    max_images: Optional[int],
) -> Dict[str, Any]:
    cfg = DEFAULTS["pose"]
    config = resolve_ckpt(cfg["task_config"], REPO_ROOT, label="pose config")
    ckpt = resolve_ckpt(cfg["task_checkpoint"], REPO_ROOT, label="pose checkpoint")
    ann_file = cfg["ann_file"]
    if not os.path.isabs(ann_file):
        ann_file = os.path.join(dataset_path, ann_file)
    if not os.path.isfile(ann_file):
        return {"error": f"missing ann_file: {ann_file}", "note": cfg.get("note")}

    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.evaluation.metrics import CocoMetric
        from mmpose.structures import merge_data_samples
    except Exception as e:
        return {"error": f"mmpose import failed: {e}", "note": cfg.get("note")}

    # Best-effort top-down pose baseline on person boxes from GT.
    # Full HigherHRNet bottom-up WholeBody is not wired here.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model = init_model(config, ckpt, device=device)
    coco = COCO(ann_file)
    images = list_coco_images(ann_file, max_images=max_images)
    img_dir = os.path.join(dataset_path, split)
    preds: List[Dict[str, Any]] = []

    for info in tqdm(images, desc="baseline:pose"):
        path = os.path.join(img_dir, info["file_name"])
        ann_ids = coco.getAnnIds(imgIds=[info["id"]], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        bboxes = []
        for a in anns:
            if "bbox" in a and a.get("num_keypoints", 0) >= 0:
                x, y, w, h = a["bbox"]
                bboxes.append([x, y, x + w, y + h])
        if not bboxes:
            continue
        try:
            results = inference_topdown(model, path, bboxes=np.asarray(bboxes, dtype=np.float32))
        except Exception as e:
            return {
                "error": f"pose inference failed: {e}",
                "note": cfg.get("note"),
                "checkpoint": ckpt,
            }
        for r in results:
            if hasattr(r, "pred_instances"):
                inst = r.pred_instances
                kpts = inst.keypoints.detach().cpu().numpy()
                scores = (
                    inst.keypoint_scores.detach().cpu().numpy()
                    if hasattr(inst, "keypoint_scores")
                    else np.ones(kpts.shape[:2], dtype=np.float32)
                )
                for person_kpt, person_score in zip(kpts, scores):
                    flat = []
                    for (x, y), s in zip(person_kpt, person_score):
                        flat.extend([float(x), float(y), float(s)])
                    preds.append(
                        {
                            "image_id": int(info["id"]),
                            "category_id": 1,
                            "keypoints": flat,
                            "score": float(np.mean(person_score)),
                        }
                    )

    if not preds:
        return {"error": "no pose predictions", "note": cfg.get("note"), "checkpoint": ckpt}

    dt = coco.loadRes(preds)
    ev = COCOeval(coco, dt, iouType="keypoints")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "task": "pose",
        "mAP-OKS": float(ev.stats[0]),
        "n_images": len(images),
        "n_preds": len(preds),
        "checkpoint": ckpt,
        "ann_file": ann_file,
        "note": cfg.get("note"),
    }


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "logs", "eval_baseline", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    wait_for_gpu(args.gpu_id, args.wait_gpu_mem_mb)

    summary: Dict[str, Any] = {
        "device": device,
        "dataset_path": args.dataset_path,
        "max_images": args.max_images,
        "tasks": {},
    }

    for task in tasks:
        if task not in DEFAULTS:
            summary["tasks"][task] = {"error": f"unknown task {task}"}
            continue
        print(f"\n==== baseline: {task} ====")
        note = DEFAULTS[task].get("note")
        if note:
            print(f"note: {note}")

        ann = DEFAULTS[task]["ann_file"]
        ann_path = ann if os.path.isabs(ann) else os.path.join(args.dataset_path, ann)
        if not os.path.isfile(ann_path) and task in ("semantic", "panoptic"):
            summary["tasks"][task] = {
                "error": f"missing annotation: {ann_path}",
                "note": note,
                "skipped": True,
            }
            print(f"SKIP: {ann_path} not found")
            continue

        try:
            if task in ("detection", "instance"):
                result = eval_detection_instance(
                    task,
                    args.dataset_path,
                    args.split,
                    device,
                    args.max_images,
                    args.score_thr,
                )
            elif task == "pose":
                result = eval_pose(args.dataset_path, args.split, device, args.max_images)
            else:
                result = {
                    "error": "not implemented for original-image baseline yet",
                    "note": note,
                    "skipped": True,
                }
        except Exception as e:
            result = {"error": str(e), "note": note}

        summary["tasks"][task] = result
        print(json.dumps(result, indent=2))

    out_json = os.path.join(out_dir, "baseline_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
