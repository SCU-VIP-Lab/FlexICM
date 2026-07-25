"""COCO-style evaluation datasets that return image + annotation paths/ids."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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


# Default annotation files relative to coco_root
TASK_ANN_FILES = {
    "detection": "annotations/instances_val2017.json",
    "instance": "annotations/instances_val2017.json",
    "semantic": "annotations/panoptic_val2017.json",  # or stuff; override in config
    "panoptic": "annotations/panoptic_val2017.json",
    "pose": "annotations/coco_wholebody_val_v1.0.json",
}
