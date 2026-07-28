"""Shared helpers for codec test / eval scripts."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

from flexicm.utils.alignment import Alignment
from flexicm.utils.train_utils import AverageMeter


def resolve_ckpt(path: str, repo_root: str, label: str = "checkpoint") -> str:
    if not path:
        raise FileNotFoundError(f"{label}: path is empty")
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    placeholder = os.path.join(os.path.dirname(path), "PLACEHOLDER")
    if os.path.isfile(placeholder) and not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label}: not ready (PLACEHOLDER still present):\n  {placeholder}\n"
            f"Expected real weights at:\n  {path}\n"
            "See checkpoints/README.md"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}: missing file: {path}")
    if os.path.basename(path) == "PLACEHOLDER" or path.endswith(".txt"):
        raise FileNotFoundError(f"{label}: refusing placeholder file: {path}")
    return path


# Eval checkpoint layout (quality_level selects the rate point 1..4):
#   TAIC:  checkpoints/taic/{task}/{quality_level}/checkpoint_best_loss.pth.tar
#   C-TAIC extension: checkpoints/ctaic/{scenario_dir}/stage2/{quality_level}/checkpoint_best_loss.pth.tar
#   C-TAIC base TAIC: checkpoints/taic/{base_task}/{quality_level}/checkpoint_best_loss.pth.tar
CTAIC_SCENARIO_DIRS = {
    "s1": "s1_det_instance",
    "s2": "s2_sem_panoptic",
    "s3": "s3_det_pose",
}


def default_taic_ckpt(task: str, quality_level: int) -> str:
    q = int(quality_level)
    return f"./checkpoints/taic/{task}/{q}/checkpoint_best_loss.pth.tar"


def default_ctaic_ckpt(scenario: str, quality_level: int) -> str:
    q = int(quality_level)
    scenario_dir = CTAIC_SCENARIO_DIRS[scenario]
    return f"./checkpoints/ctaic/{scenario_dir}/stage2/{q}/checkpoint_best_loss.pth.tar"


def default_base_taic_ckpt(base_task: str, quality_level: int) -> str:
    q = int(quality_level)
    return f"./checkpoints/taic/{base_task}/{q}/checkpoint_best_loss.pth.tar"


def crop_feature_to_image(h: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    """Crop decoded feature h (H/4, W/4 of padded input) to original image size / 4."""
    H, W = image_hw
    return h[..., : H // 4, : W // 4]


def pad_for_codec(images: torch.Tensor, divisor: int = 256, device=None):
    align = Alignment(divisor=divisor, mode="pad", padding_mode="constant")
    if device is not None:
        align = align.to(device)
    return align.align(images), align


def likelihood_bpp(likelihoods: Dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    import math

    return sum(
        (torch.log(lik).sum() / (-math.log(2) * num_pixels))
        for lik in likelihoods.values()
    )


def actual_bitstream_bpp(strings, num_pixels: int) -> float:
    """Estimate bpp from CompressAI byte strings: [[y_bytes...], [z_bytes...]]."""
    total_bits = 0
    for group in strings:
        for s in group:
            if isinstance(s, (bytes, bytearray)):
                total_bits += len(s) * 8
            elif torch.is_tensor(s):
                total_bits += int(s.numel() * s.element_size() * 8)
            else:
                total_bits += len(s) * 8
    return total_bits / float(num_pixels)


@torch.no_grad()
def test_taic_loader(
    model,
    teacher,
    loader,
    criterion,
    device,
    align_divisor: int = 256,
    run_actual_bpp: bool = False,
    max_batches: Optional[int] = None,
    log_every: int = 50,
):
    """Run codec test: likelihood bpp + feature distortion (+ optional real bpp)."""
    model.eval()
    teacher.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion", "actual_bpp")}

    if run_actual_bpp:
        model.update(force=True)

    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, total=total, desc="codec eval", leave=True)
    for i, images in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        N, _, H, W = images.shape
        num_pixels = N * H * W

        # Keep codec + teacher on the padded grid so Swin always sees a
        # patch/window-divisible size. bpp still uses the original pixel count.
        x, _ = pad_for_codec(images, divisor=align_divisor, device=device)
        out = model(x)

        gt = teacher.gt_features(x)
        pred = teacher.pred_features(out["h"])
        stats = criterion(out, pred, gt, num_pixels=num_pixels)

        meters["loss"].update(stats["loss"].item(), n=N)
        meters["bpp"].update(stats["bpp"].item(), n=N)
        meters["distortion"].update(stats["distortion"].item(), n=N)

        if run_actual_bpp:
            try:
                enc = model.compress(x)
                dec = model.decompress(
                    enc["strings"], enc["shape"], x_size=(x.shape[2], x.shape[3])
                )
                abpp = actual_bitstream_bpp(enc["strings"], num_pixels)
                meters["actual_bpp"].update(abpp, n=N)
                # sanity: decoded h spatial size
                _ = dec["h"]
            except Exception as e:
                if i == 0:
                    print(f"[warn] actual bpp / compress-decompress failed: {e}")

    result = {
        "bpp": meters["bpp"].avg,
        "distortion": meters["distortion"].avg,
        "loss": meters["loss"].avg,
        "num_batches": meters["bpp"].count,
    }
    if run_actual_bpp and meters["actual_bpp"].count > 0:
        result["actual_bpp"] = meters["actual_bpp"].avg
    return result


@torch.no_grad()
def test_ctaic_loader(
    ext_model,
    base_model,
    teacher,
    loader,
    criterion,
    device,
    use_condition: bool = True,
    align_divisor: int = 256,
    run_actual_bpp: bool = False,
    max_batches: Optional[int] = None,
    log_every: int = 50,
):
    """Codec test for C-TAIC; bpp is extension-layer only (paper Sec.IV.E.2)."""
    ext_model.eval()
    base_model.eval()
    teacher.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "distortion", "actual_bpp")}

    if run_actual_bpp:
        ext_model.update(force=True)

    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, total=total, desc="codec eval", leave=True)
    for i, images in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        N, _, H, W = images.shape
        num_pixels = N * H * W

        # Keep codec + teacher on the padded grid (same rationale as TAIC test).
        x, _ = pad_for_codec(images, divisor=align_divisor, device=device)
        y_b = None
        if use_condition:
            base_out = base_model(x)
            y_b = base_out["y_hat"]

        out = ext_model(x, y_b_hat=y_b, use_condition=use_condition and y_b is not None)

        gt = teacher.gt_features(x)
        pred = teacher.pred_features(out["h"])
        stats = criterion(out, pred, gt, num_pixels=num_pixels)

        meters["loss"].update(stats["loss"].item(), n=N)
        meters["bpp"].update(stats["bpp"].item(), n=N)
        meters["distortion"].update(stats["distortion"].item(), n=N)

        if run_actual_bpp:
            try:
                enc = ext_model.compress(
                    x, y_b_hat=y_b, use_condition=use_condition and y_b is not None
                )
                dec = ext_model.decompress(
                    enc["strings"],
                    enc["shape"],
                    x_size=(x.shape[2], x.shape[3]),
                    y_b_hat=y_b,
                    use_condition=use_condition and y_b is not None,
                )
                abpp = actual_bitstream_bpp(enc["strings"], num_pixels)
                meters["actual_bpp"].update(abpp, n=N)
                _ = dec["h"]
            except Exception as e:
                if i == 0:
                    print(f"[warn] actual bpp / compress-decompress failed: {e}")

    result = {
        "bpp": meters["bpp"].avg,
        "distortion": meters["distortion"].avg,
        "loss": meters["loss"].avg,
        "num_batches": meters["bpp"].count,
        "use_condition": use_condition,
    }
    if run_actual_bpp and meters["actual_bpp"].count > 0:
        result["actual_bpp"] = meters["actual_bpp"].avg
    return result
