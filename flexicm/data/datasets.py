"""Datasets and preprocessing aligned with task-network training."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Official task-network keep-ratio geometry (before codec pad÷256).
# Det/inst/panoptic: MMDet Resize(scale=(1333, 800), keep_ratio=True)
# Pose: HigherHRNet BottomupResize expand, short edge 512
# Semantic: ADE UPerNet test short 512 / long ≤2048
TASK_ALIGN_DEFAULTS = {
    "detection": {"short_edge": 800, "max_long_side": 1333},
    "det": {"short_edge": 800, "max_long_side": 1333},
    "object_detection": {"short_edge": 800, "max_long_side": 1333},
    "instance": {"short_edge": 800, "max_long_side": 1333},
    "instance_seg": {"short_edge": 800, "max_long_side": 1333},
    "instance_segmentation": {"short_edge": 800, "max_long_side": 1333},
    "panoptic": {"short_edge": 800, "max_long_side": 1333},
    "panoptic_seg": {"short_edge": 800, "max_long_side": 1333},
    "panoptic_segmentation": {"short_edge": 800, "max_long_side": 1333},
    "pose": {"short_edge": 512, "max_long_side": None},
    "pose_estimation": {"short_edge": 512, "max_long_side": None},
    "semantic": {"short_edge": 512, "max_long_side": 2048},
    "semantic_seg": {"short_edge": 512, "max_long_side": 2048},
    "semantic_segmentation": {"short_edge": 512, "max_long_side": 2048},
}

_MISSING = object()


class LimitLongSide:
    """If the longer edge exceeds ``max_long_side``, scale the whole image down.

    Applied after short-edge expand resize. Short edge may then fall below
    ``short_edge``; that is intentional when capping GPU memory.
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


def canonicalize_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    return str(task).strip().lower()


def task_align_geom(
    task: Optional[str],
    short_edge: Optional[int] = None,
    max_long_side: Any = _MISSING,
) -> tuple:
    """Resolve (short_edge, max_long_side) for a task.

    ``max_long_side=_MISSING`` → use task default (may be ``None`` = no cap).
    Explicit ``None`` keeps no long-edge cap.
    """
    key = canonicalize_task(task)
    defaults = TASK_ALIGN_DEFAULTS.get(key or "", {})
    se = int(short_edge) if short_edge is not None else defaults.get("short_edge")
    if se is None:
        se = 256
    if max_long_side is _MISSING:
        ml = defaults.get("max_long_side", None)
    else:
        ml = None if max_long_side is None else int(max_long_side)
    return int(se), ml


def build_expand_transform(
    short_edge: int = 256,
    max_long_side: Optional[int] = None,
) -> Callable:
    """Keep-ratio short-edge resize (+ optional long-edge cap), then ToTensor.

    Matches MMDet ``Resize(short, max_size=long)`` / pose expand semantics.
    Variable HxW are batched via ``collate_expand_pad`` (pad to ÷256) or
    ``pad_for_codec`` at eval time.
    """
    size = int(short_edge)
    ops: List[Callable] = [
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]
    if max_long_side is not None:
        # torchvision Resize(max_size=...) requires max_size > size; use our op
        # so pose/train caps like max_long_side == short_edge still work.
        ops.append(LimitLongSide(int(max_long_side)))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_train_transform(
    patch_size: int = 256,
    max_long_side: Optional[int] = None,
) -> Callable:
    """Backward-compatible alias: ``patch_size`` = short edge (expand)."""
    return build_expand_transform(short_edge=patch_size, max_long_side=max_long_side)


def build_task_aligned_transform(
    task: Optional[str] = None,
    *,
    short_edge: Optional[int] = None,
    max_long_side: Any = _MISSING,
    eval_size: Optional[int] = None,
    align_to_task: bool = True,
) -> Callable:
    """Preprocess aligned with the official task-network test geometry.

    - ``eval_size`` set → legacy anisotropic square resize (not recommended)
    - ``align_to_task=False`` → native resolution ``ToTensor`` only
    - else short-edge / max-long keep-ratio (task defaults or overrides)
    """
    if eval_size is not None:
        return build_test_transform(eval_size=eval_size)
    if not align_to_task:
        return transforms.ToTensor()
    se, ml = task_align_geom(task, short_edge=short_edge, max_long_side=max_long_side)
    return build_expand_transform(short_edge=se, max_long_side=ml)


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
