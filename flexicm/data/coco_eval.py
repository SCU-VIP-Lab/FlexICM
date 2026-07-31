"""COCO-style evaluation datasets that return image + annotation paths/ids."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class COCOEvalDataset(Dataset):
    """COCO val images with annotation ids for metric evaluation.

    Returns a dict:
      image: FloatTensor CxHxW in [0,1]
      image_id: int
      file_name: str
      height, width: int
      path: str
    """

    def __init__(
        self,
        coco_root: str,
        ann_file: str,
        image_prefix: str = "val2017",
        transform=None,
    ):
        self.coco_root = coco_root
        self.image_dir = os.path.join(coco_root, image_prefix)
        self.ann_file = ann_file if os.path.isabs(ann_file) else os.path.join(coco_root, ann_file)
        self.transform = transform or transforms.ToTensor()

        with open(self.ann_file) as f:
            coco = json.load(f)
        self.images: List[Dict[str, Any]] = sorted(coco["images"], key=lambda x: x["id"])
        self.categories = coco.get("categories", [])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.images[index]
        path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return {
            "image": tensor,
            "image_id": int(info["id"]),
            "file_name": info["file_name"],
            "height": int(info["height"]),
            "width": int(info["width"]),
            "path": path,
        }


def coco_eval_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that keeps variable-size images as a list (batch_size usually 1)."""
    return {
        "images": [b["image"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "file_names": [b["file_name"] for b in batch],
        "heights": [b["height"] for b in batch],
        "widths": [b["width"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


def resolve_panoptic_gt_folder(
    ann_file: str,
    panoptic_gt_folder: Optional[str] = None,
) -> Optional[str]:
    """Resolve COCO panoptic PNG folder next to panoptic_val2017.json."""
    if panoptic_gt_folder and os.path.isdir(panoptic_gt_folder):
        return panoptic_gt_folder
    default = os.path.join(os.path.dirname(ann_file), "panoptic_val2017")
    return default if os.path.isdir(default) else None


def build_panoptic_semantic_gt_loader(
    ann_file: str,
    gt_folder: str,
) -> Tuple[Callable[[int], Any], int]:
    """Build image_id -> semantic label-map loader from COCO panoptic GT.

    Returns:
      loader(image_id) -> HxW numpy int64 label map on contiguous category ids.
      num_classes -> number of categories found in the panoptic json.
    """
    import numpy as np
    from PIL import Image

    with open(ann_file) as f:
        panoptic = json.load(f)

    annotations = {
        int(ann["image_id"]): ann for ann in panoptic.get("annotations", [])
    }
    categories = panoptic.get("categories", [])
    cat_to_idx = {int(cat["id"]): idx for idx, cat in enumerate(categories)}

    @lru_cache(maxsize=64)
    def _load(image_id: int):
        ann = annotations[int(image_id)]
        png_path = os.path.join(gt_folder, ann["file_name"])
        rgb = np.array(Image.open(png_path).convert("RGB"), dtype=np.int64)
        seg_ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]

        # 255 as ignore label for segments not covered by segments_info.
        label = np.full(seg_ids.shape, 255, dtype=np.int64)
        for seg in ann.get("segments_info", []):
            seg_id = int(seg["id"])
            cat_id = int(seg["category_id"])
            idx = cat_to_idx.get(cat_id, 255)
            label[seg_ids == seg_id] = idx
        return label

    return _load, len(categories)


def enrich_panoptic_finalize_kwargs(
    task: str,
    ann_file: str,
    finalize_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach panoptic GT folder and optional semantic GT loader for metric eval."""
    if os.path.basename(ann_file) != "panoptic_val2017.json":
        return finalize_kwargs

    panoptic_gt_folder = resolve_panoptic_gt_folder(
        ann_file, finalize_kwargs.get("gt_folder")
    )
    if not panoptic_gt_folder:
        print(
            f"[metric] panoptic GT folder missing for {ann_file}\n"
            f"  semantic mIoU / panoptic PQ may be NaN"
        )
        return finalize_kwargs

    finalize_kwargs = dict(finalize_kwargs)
    finalize_kwargs["gt_folder"] = panoptic_gt_folder
    if task == "semantic":
        gt_seg_loader, num_classes = build_panoptic_semantic_gt_loader(
            ann_file, panoptic_gt_folder
        )
        finalize_kwargs["gt_seg_loader"] = gt_seg_loader
        finalize_kwargs["num_classes"] = num_classes
        # Default ADE→COCO 38-class allowlist unless caller overrides.
        if not finalize_kwargs.get("eval_classes_file") and not finalize_kwargs.get(
            "eval_class_names"
        ):
            default_classes = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs",
                "eval",
                "semantic_ade_coco_eval_classes.json",
            )
            if os.path.isfile(default_classes):
                finalize_kwargs["eval_classes_file"] = default_classes
        print(
            f"[metric] semantic GT loader from panoptic GT:\n"
            f"  gt_folder={panoptic_gt_folder}\n"
            f"  num_classes={num_classes}\n"
            f"  eval_classes_file={finalize_kwargs.get('eval_classes_file')}"
        )
    elif task == "panoptic":
        print(f"[metric] panoptic GT folder resolved: {panoptic_gt_folder}")
    return finalize_kwargs


# Default annotation files relative to coco_root
TASK_ANN_FILES = {
    "detection": "annotations/instances_val2017.json",
    "instance": "annotations/instances_val2017.json",
    "semantic": "annotations/panoptic_val2017.json",  # or stuff; override in config
    "panoptic": "annotations/panoptic_val2017.json",
    "pose": "annotations/person_keypoints_val2017.json",
}
