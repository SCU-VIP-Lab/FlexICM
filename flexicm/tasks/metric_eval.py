"""End-to-end codec + task-network metric evaluation loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from flexicm.utils.codec_test import pad_for_codec
from flexicm.tasks.metric_runners import TaskMetricRunner


def normalize_pose_ms_scales(scales: Optional[Sequence[float]]) -> List[float]:
    """Return pose multi-scale factors with ``1.0`` first (tags / decode base)."""
    if not scales:
        return [1.0]
    vals = [float(s) for s in scales]
    if any(s <= 0 for s in vals):
        raise ValueError(f"pose_ms_scales must be > 0, got {scales}")
    if not any(abs(s - 1.0) < 1e-8 for s in vals):
        vals = [1.0] + vals
    ordered: List[float] = []
    seen = set()
    for s in [1.0] + [x for x in vals if abs(x - 1.0) >= 1e-8]:
        key = round(s, 6)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(s)
    return ordered


def _codec_forward(codec, x, use_condition: bool, base_codec):
    if use_condition and base_codec is not None:
        y_b = base_codec(x)["y_hat"]
        return codec(x, y_b_hat=y_b, use_condition=True)
    if hasattr(codec, "forward") and use_condition is False and base_codec is None:
        return codec(x)
    if hasattr(codec, "forward"):
        try:
            return codec(x, y_b_hat=None, use_condition=False)
        except TypeError:
            return codec(x)
    return codec(x)


@torch.no_grad()
def run_task_metric_eval(
    codec,
    runner: TaskMetricRunner,
    loader,
    device: str,
    ann_file: str,
    use_condition: bool = False,
    base_codec=None,
    align_divisor: int = 256,
    max_batches: Optional[int] = None,
    finalize_kwargs: Optional[Dict[str, Any]] = None,
    pose_ms_scales: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """For each image: codec -> h -> truncated task net -> accumulate -> metrics.

    For pose, optional ``pose_ms_scales`` (e.g. ``[1.0, 2.0, 0.5]``) runs the
    codec at multiple input resolutions, merges heatmaps in the pose head, then
    decodes keypoints once (MMPose HigherHRNet-style MS, without flip-TTA).
    """
    codec.eval()
    predictions: List[Any] = []
    ms_scales = normalize_pose_ms_scales(pose_ms_scales)
    use_pose_ms = len(ms_scales) > 1 and hasattr(runner, "predict_from_ms_h")
    if use_pose_ms:
        print(f"[metric] pose multi-scale via codec: {ms_scales}")

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # Support both plain image batches and COCOEval collate dicts
        if isinstance(batch, dict) and "images" in batch:
            images_list = batch["images"]
            metas = []
            for j in range(len(images_list)):
                metas.append(
                    dict(
                        image_id=batch["image_ids"][j],
                        height=batch["heights"][j],
                        width=batch["widths"][j],
                        path=batch["paths"][j],
                        file_name=batch["file_names"][j],
                    )
                )
        else:
            # Tensor batch Bx3xHxW without coco ids — skip metric (needs image_id)
            raise RuntimeError(
                "Task-metric eval requires COCOEvalDataset + coco_eval_collate "
                "(image_id / height / width)."
            )

        for image, meta in zip(images_list, metas):
            image = image.unsqueeze(0).to(device)
            _, _, H, W = image.shape
            ori_h = int(meta["height"])
            ori_w = int(meta["width"])
            # If the image was resized (e.g. eval_size=256), boxes must be
            # mapped back with the correct scale_factor for COCO mAP.
            scale_w = float(W) / float(ori_w) if ori_w > 0 else 1.0
            scale_h = float(H) / float(ori_h) if ori_h > 0 else 1.0
            meta = dict(meta)
            meta["scale_factor"] = (scale_w, scale_h)

            if use_pose_ms:
                ms_pack = []
                for s in ms_scales:
                    if abs(s - 1.0) < 1e-8:
                        img_s = image
                    else:
                        nh = max(1, int(round(H * s)))
                        nw = max(1, int(round(W * s)))
                        img_s = F.interpolate(
                            image, size=(nh, nw), mode="bilinear", align_corners=False
                        )
                    x, _ = pad_for_codec(img_s, divisor=align_divisor, device=device)
                    out = _codec_forward(codec, x, use_condition, base_codec)
                    ms_pack.append(
                        {
                            "h": out["h"],
                            "image_tensor": x,
                            "pad_height": int(x.shape[-2]),
                            "pad_width": int(x.shape[-1]),
                            "scale": float(s),
                        }
                    )
                # Decode / COCO mapping use the base scale (1.0) canvas.
                meta["pad_height"] = ms_pack[0]["pad_height"]
                meta["pad_width"] = ms_pack[0]["pad_width"]
                meta["image_tensor"] = ms_pack[0]["image_tensor"]
                pred = runner.predict_from_ms_h(ms_pack, meta)
            else:
                x, _ = pad_for_codec(image, divisor=align_divisor, device=device)
                out = _codec_forward(codec, x, use_condition, base_codec)
                h = out["h"]
                meta["pad_height"] = int(x.shape[-2])
                meta["pad_width"] = int(x.shape[-1])
                # Keep the padded RGB tensor for task nets that need a stem
                # (e.g. HRNet multi-branch pose) while still injecting codec h.
                meta["image_tensor"] = x
                pred = runner.predict_from_h(h, meta)
            predictions.append(pred)

        if i % 20 == 0:
            print(f"[metric] processed batch {i}/{len(loader)}")

    metrics = runner.finalize(predictions, ann_file, **(finalize_kwargs or {}))
    if use_pose_ms:
        metrics = dict(metrics)
        metrics["pose_ms_scales"] = list(ms_scales)
    return metrics
