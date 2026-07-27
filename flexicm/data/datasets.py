"""Datasets and preprocessing aligned with task-network training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(patch_size: int = 256) -> Callable:
    """Codec training crops; keep RGB float in [0, 1] (normalization inside teacher)."""
    return transforms.Compose(
        [
            transforms.Resize(patch_size),
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_test_transform(eval_size: Optional[int] = None) -> Callable:
    """Eval preprocess. If ``eval_size`` is set (e.g. 256), force HxW = size×size."""
    if eval_size is None:
        return transforms.ToTensor()
    return transforms.Compose(
        [
            transforms.Resize((int(eval_size), int(eval_size))),
            transforms.ToTensor(),
        ]
    )


class ImageFolderDataset(Dataset):
    """Generic image folder (recursive) for codec training."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform: Optional[Callable] = None, list_file: Optional[str] = None):
        self.root = root
        self.transform = transform
        if list_file and os.path.isfile(list_file):
            with open(list_file) as f:
                rels = [ln.strip() for ln in f if ln.strip()]
            self.files = [os.path.join(root, r) if not os.path.isabs(r) else r for r in rels]
        else:
            self.files = []
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if Path(fn).suffix.lower() in self.IMG_EXTS:
                        self.files.append(os.path.join(dirpath, fn))
            self.files.sort()
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class COCOImageDataset(ImageFolderDataset):
    """COCO images for detection / instance / semantic / panoptic codec training."""

    def __init__(self, coco_root: str, split: str = "train2017", transform=None, list_file=None):
        image_dir = os.path.join(coco_root, split)
        super().__init__(image_dir, transform=transform, list_file=list_file)


class COCOWholeBodyImageDataset(ImageFolderDataset):
    """COCO-WholeBody uses the same COCO images; annotations differ at eval time."""

    def __init__(self, coco_root: str, split: str = "train2017", transform=None, list_file=None):
        image_dir = os.path.join(coco_root, split)
        super().__init__(image_dir, transform=transform, list_file=list_file)


def collate_keep(batch):
    return torch.stack(batch, dim=0)
