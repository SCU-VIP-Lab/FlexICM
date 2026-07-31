"""End-to-end codec + task-network metric evaluation loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from flexicm.utils.codec_test import crop_feature_to_image, pad_for_codec
from flexicm.tasks.metric_runners import TaskMetricRunner, build_metric_runner


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
) -> Dict[str, float]:
    """For each image: codec -> h -> truncated task net -> accumulate -> metrics."""
    codec.eval()
    predictions: List[Any] = []

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
            x, _ = pad_for_codec(image, divisor=align_divisor, device=device)

            y_b = None
            if use_condition and base_codec is not None:
                y_b = base_codec(x)["y_hat"]
                out = codec(x, y_b_hat=y_b, use_condition=True)
            elif hasattr(codec, "forward") and use_condition is False and base_codec is None:
                out = codec(x)
            else:
                # CTAIC without condition
                if hasattr(codec, "forward"):
                    try:
                        out = codec(x, y_b_hat=None, use_condition=False)
                    except TypeError:
                        out = codec(x)
                else:
                    out = codec(x)

            h = out["h"]
            meta = dict(meta)
            meta["pad_height"] = int(x.shape[-2])
            meta["pad_width"] = int(x.shape[-1])
            meta["scale_factor"] = (scale_w, scale_h)
            # Keep the padded RGB tensor for task nets that need a stem
            # (e.g. HRNet multi-branch pose) while still injecting codec h.
            meta["image_tensor"] = x
            pred = runner.predict_from_h(h, meta)
            predictions.append(pred)

        if i % 20 == 0:
            print(f"[metric] processed batch {i}/{len(loader)}")

    metrics = runner.finalize(predictions, ann_file, **(finalize_kwargs or {}))
    return metrics
