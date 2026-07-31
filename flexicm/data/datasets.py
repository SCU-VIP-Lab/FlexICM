"""Datasets and preprocessing aligned with task-network training."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LimitLongSide:
    """If the longer edge exceeds ``max_long_side``, scale the whole image down.

    Applied after short-edge expand resize. Short edge may then fall below
    ``patch_size``; that is intentional to cap GPU memory.
    """

    def __init__(self, max_long_side: int):
        self.max_long_side = int(max_long_side)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        long = max(w, h)
        if long <= self.max_long_side:
            return img
        scale = self.max_long_side / float(long)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), resample=Image.BILINEAR)


def build_train_transform(
    patch_size: int = 256,
    max_long_side: Optional[int] = None,
) -> Callable:
    """Deterministic train preprocess matching HigherHRNet BottomupResize(expand).

    - Scale by the **shorter** side to ``patch_size``, keep aspect ratio
    - Keep the **full** image (longer side may exceed ``patch_size``)
    - If ``max_long_side`` is set, shrink again when the long edge exceeds it
    - No crop / flip / anisotropic stretch

    Variable HxW are batched via ``collate_expand_pad`` (pad to ÷256).
    """
    size = int(patch_size)
    ops: List[Callable] = [
        # torchvision: int size → resize shorter edge to ``size``, keep ratio
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]
    if max_long_side is not None:
        ops.append(LimitLongSide(int(max_long_side)))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_test_transform(eval_size: Optional[int] = None) -> Callable:
    """Eval preprocess. If ``eval_size`` is set (e.g. 256), force HxW = size×size."""
    if eval_size is None:
        return transforms.ToTensor()
    return transforms.Compose(
        [
            transforms.Resize(
                (int(eval_size), int(eval_size)),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )


def _ceil_to_multiple(v: int, base: int) -> int:
    return int(math.ceil(v / float(base)) * base)


def collate_expand_pad(
    batch: Sequence[torch.Tensor],
    divisor: int = 256,
) -> torch.Tensor:
    """Stack expand-resized images by center-padding to a shared ÷divisor canvas.

    Mirrors pose ``BottomupResize(expand)`` + pad: content stays centered, borders 0.
    """
    assert len(batch) > 0
    assert all(isinstance(x, torch.Tensor) and x.ndim == 3 for x in batch)
    max_h = max(int(x.shape[-2]) for x in batch)
    max_w = max(int(x.shape[-1]) for x in batch)
    canvas_h = _ceil_to_multiple(max_h, divisor)
    canvas_w = _ceil_to_multiple(max_w, divisor)
    c = int(batch[0].shape[0])
    out = batch[0].new_zeros((len(batch), c, canvas_h, canvas_w))
    for i, x in enumerate(batch):
        _, h, w = x.shape
        top = (canvas_h - h) // 2
        left = (canvas_w - w) // 2
        out[i, :, top : top + h, left : left + w] = x
    return out


def collate_keep(batch):
    return torch.stack(batch, dim=0)


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
