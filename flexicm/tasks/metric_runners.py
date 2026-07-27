"""Task-network metric runners: load official checkpoints and evaluate from codec feature h.

Paper flow (Sec.III.A):
  codec -> h (H/4 x W/4 x C) -> truncated task network (from Stage 2 / FPN) -> task output
  then compute mAP-bbox / mAP-mask / mIoU / PQ / mAP-OKS.

Requires optional packages:
  pip install pycocotools
  mim install mmdet mmsegmentation mmpose   # plus mmengine mmcv
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskMetricRunner(ABC):
    """Unified interface for end-task evaluation from decoded feature h."""

    metric_name: str = "metric"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self, config_path: str, checkpoint_path: str) -> None:
        ...

    @abstractmethod
    def predict_from_h(
        self,
        h: torch.Tensor,
        img_meta: Dict[str, Any],
    ) -> Any:
        """Run truncated task net starting from feature h."""
        ...

    @abstractmethod
    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        """Aggregate predictions vs GT annotations into scalar metrics."""
        ...


def _require_mmdet():
    try:
        import mmdet  # noqa: F401
        from mmdet.apis import init_detector
    except ImportError as e:
        raise ImportError(
            "Full task-metric evaluation requires MMDetection.\n"
            "  pip install -U openmim && mim install mmengine mmcv mmdet"
        ) from e
    return init_detector


def _require_mmseg():
    try:
        from mmseg.apis import init_model
    except ImportError as e:
        raise ImportError(
            "Semantic segmentation metric evaluation requires MMSegmentation.\n"
            "  mim install mmsegmentation"
        ) from e
    return init_model


def _require_mmpose():
    try:
        from mmpose.apis import init_model
    except ImportError as e:
        raise ImportError(
            "Pose metric evaluation requires MMPose.\n"
            "  mim install mmpose"
        ) from e
    return init_model


def swin_feats_from_h(backbone: nn.Module, h: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Treat h as Swin F1 (stage-0 output) and run remaining stages.

    Compatible with:
      - MMDet 3.x SwinTransformer (token + hw_shape API, ``stages``)
      - older Swin / MMSeg variants that expose ``stages`` / ``layers``
    """
    stages = None
    for name in ("stages", "layers"):
        if hasattr(backbone, name):
            stages = getattr(backbone, name)
            break
    if stages is None:
        raise RuntimeError("Backbone has no stages/layers; cannot inject h as F1")

    # h is NCHW (B,C,H,W) == F1
    if h.dim() != 4:
        raise ValueError(f"expected NCHW h, got shape {tuple(h.shape)}")
    B, C, H, W = h.shape
    outs = [h]

    # Detect mmdet3-style SwinBlockSequence: forward(x, hw_shape)
    import inspect

    try:
        needs_hw = "hw_shape" in inspect.signature(stages[0].forward).parameters
    except (TypeError, ValueError):
        needs_hw = False

    if needs_hw:
        # tokens: (B, H*W, C)
        x = h.flatten(2).transpose(1, 2).contiguous()
        hw_shape = (H, W)

        # F1 is stage-0 block output (pre-downsample). Feed stage-0 downsample
        # then run stages 1..N-1, mirroring SwinTransformer.forward.
        if getattr(stages[0], "downsample", None) is not None:
            x, hw_shape = stages[0].downsample(x, hw_shape)

        for i in range(1, len(stages)):
            x, hw_shape, out, out_hw_shape = stages[i](x, hw_shape)
            norm_name = f"norm{i}"
            if hasattr(backbone, norm_name):
                out = getattr(backbone, norm_name)(out)
            feat_dim = out.shape[-1]
            out = (
                out.view(B, out_hw_shape[0], out_hw_shape[1], feat_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            outs.append(out)

        # Optionally normalize F1 with norm0 for consistency with extract_feat
        if hasattr(backbone, "norm0"):
            f1 = h.flatten(2).transpose(1, 2).contiguous()
            f1 = backbone.norm0(f1)
            outs[0] = (
                f1.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            )
        return tuple(outs)

    # Legacy path (timm / older mmdet): stage modules accept NCHW / plain tensors
    x = h
    for i in range(1, len(stages)):
        x = stages[i](x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() == 4 and x.shape[1] < x.shape[-1] and x.shape[-1] in (
            96, 128, 192, 256, 384, 512, 768, 1024
        ):
            x = x.permute(0, 3, 1, 2).contiguous()
        outs.append(x)

    if hasattr(backbone, "num_features") or hasattr(backbone, "out_indices"):
        norm_outs = []
        for i, out in enumerate(outs):
            norm_name = f"norm{i}"
            if hasattr(backbone, norm_name):
                nchw = out
                norm = getattr(backbone, norm_name)
                try:
                    y = nchw.permute(0, 2, 3, 1)
                    y = norm(y)
                    nchw = y.permute(0, 3, 1, 2).contiguous()
                except Exception:
                    nchw = out
                norm_outs.append(nchw)
            else:
                norm_outs.append(out)
        return tuple(norm_outs)
    return tuple(outs)


class DetectionMetricRunner(TaskMetricRunner):
    """Cascade Mask R-CNN + Swin-B (official zoo) — mAP-bbox / mAP-mask."""

    def __init__(self, device: str = "cuda", with_mask: bool = False):
        super().__init__(device)
        self.with_mask = with_mask
        self.metric_name = "mAP-mask" if with_mask else "mAP-bbox"
        self._results: List[Dict] = []

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_detector = _require_mmdet()
        self.model = init_detector(config_path, checkpoint_path, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        assert self.model is not None
        # h: 1xCx(H/4)x(W/4) — preferably on the padded codec grid
        backbone = self.model.backbone
        feats = swin_feats_from_h(backbone, h)
        if hasattr(self.model, "neck") and self.model.neck is not None:
            feats = self.model.neck(feats)

        # Build a minimal img_metas / data_samples for mmdet 3.x or 2.x
        ori_h, ori_w = int(img_meta["height"]), int(img_meta["width"])
        # If h comes from a padded codec input, prefer those spatial sizes so
        # FPN strides line up; boxes are rescaled back to ori_shape.
        pad_h = int(img_meta.get("pad_height", h.shape[-2] * 4))
        pad_w = int(img_meta.get("pad_width", h.shape[-1] * 4))
        sf = img_meta.get("scale_factor", (1.0, 1.0))
        if isinstance(sf, (int, float)):
            scale_factor = (float(sf), float(sf))
        else:
            scale_factor = (float(sf[0]), float(sf[1]))
        try:
            # MMDet 3.x style
            from mmdet.structures import DetDataSample

            data_sample = DetDataSample()
            data_sample.set_metainfo(
                dict(
                    img_shape=(pad_h, pad_w),
                    ori_shape=(ori_h, ori_w),
                    pad_shape=(pad_h, pad_w),
                    scale_factor=scale_factor,
                    img_id=img_meta.get("image_id"),
                )
            )
            rpn_results_list = self.model.rpn_head.predict(feats, [data_sample], rescale=False)
            results_list = self.model.roi_head.predict(
                feats, rpn_results_list, [data_sample], rescale=True
            )
            pred = results_list[0]
            # mmdet 3.x roi_head.predict may return DetDataSample or InstanceData
            inst = pred.pred_instances if hasattr(pred, "pred_instances") else pred
            out = {
                "image_id": img_meta["image_id"],
                "bboxes": inst.bboxes.detach().cpu(),
                "scores": inst.scores.detach().cpu(),
                "labels": inst.labels.detach().cpu(),
            }
            if self.with_mask and hasattr(inst, "masks") and inst.masks is not None:
                out["masks"] = inst.masks.to_ndarray() if hasattr(inst.masks, "to_ndarray") else inst.masks.detach().cpu()
            return out
        except Exception:
            # Fallback MMDet 2.x
            img_metas = [
                dict(
                    img_shape=(pad_h, pad_w, 3),
                    ori_shape=(ori_h, ori_w, 3),
                    pad_shape=(pad_h, pad_w, 3),
                    scale_factor=scale_factor,
                    flip=False,
                )
            ]
            proposal_list = self.model.rpn_head.simple_test_rpn(feats, img_metas)
            det_results = self.model.roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=True
            )
            # det_results: list of (bboxes_per_class) or (bboxes, segm)
            return {"image_id": img_meta["image_id"], "raw": det_results[0]}

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import numpy as np

        coco_gt = COCO(ann_file)
        coco_results = []
        for pred in predictions:
            if pred is None:
                continue
            if "raw" in pred:
                # mmdet 2.x format: list[ndarray(n,5)] per class
                raw = pred["raw"]
                bbox_results = raw[0] if isinstance(raw, tuple) else raw
                for label, bboxes in enumerate(bbox_results):
                    for row in bboxes:
                        x1, y1, x2, y2, score = row[:5]
                        coco_results.append(
                            {
                                "image_id": int(pred["image_id"]),
                                "category_id": int(coco_gt.getCatIds()[label])
                                if label < len(coco_gt.getCatIds())
                                else int(label + 1),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(score),
                            }
                        )
                continue

            bboxes = pred["bboxes"].numpy()
            scores = pred["scores"].numpy()
            labels = pred["labels"].numpy()
            cat_ids = coco_gt.getCatIds()
            for box, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                cat_id = int(cat_ids[int(label)]) if int(label) < len(cat_ids) else int(label) + 1
                coco_results.append(
                    {
                        "image_id": int(pred["image_id"]),
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

        if not coco_results:
            return {self.metric_name: 0.0}

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = {"mAP-bbox": float(coco_eval.stats[0])}

        if self.with_mask:
            # Mask eval requires segmentation results in COCO format; if unavailable, skip
            try:
                coco_eval_m = COCOeval(coco_gt, coco_dt, iouType="segm")
                coco_eval_m.evaluate()
                coco_eval_m.accumulate()
                coco_eval_m.summarize()
                metrics["mAP-mask"] = float(coco_eval_m.stats[0])
            except Exception as e:
                metrics["mAP-mask_error"] = str(e)
        return metrics


class SemanticMetricRunner(TaskMetricRunner):
    """UPerNet (MMSeg) — metric: mIoU."""

    metric_name = "mIoU"

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_model = _require_mmseg()
        self.model = init_model(config_path, checkpoint_path, device=self.device)
        self.model.eval()
        self._preds = []

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        assert self.model is not None
        backbone = self.model.backbone
        feats = swin_feats_from_h(backbone, h)
        seg_logits = self.model.decode_head(feats)
        if isinstance(seg_logits, (tuple, list)):
            seg_logits = seg_logits[0]
        H, W = int(img_meta["height"]), int(img_meta["width"])
        seg = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = seg.argmax(dim=1)[0].detach().cpu().numpy()
        return {"image_id": img_meta["image_id"], "seg": pred, "path": img_meta.get("path")}

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        """Compute mIoU if GT semantic maps are provided via kwargs['gt_dir'] or panoptic conversion.

        For a minimal working path, expects kwargs['gt_seg_loader'](image_id)->HxW label map.
        """
        gt_loader = kwargs.get("gt_seg_loader")
        if gt_loader is None:
            return {
                "mIoU": float("nan"),
                "note": "Provide gt_seg_loader or use panoptic stuff GT to compute mIoU",
            }

        import numpy as np

        num_classes = int(kwargs.get("num_classes", 133))
        intersect = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)
        for pred in predictions:
            gt = gt_loader(pred["image_id"])
            pr = pred["seg"]
            if gt.shape != pr.shape:
                # nearest resize pred already at image size; skip mismatch
                continue
            for c in range(num_classes):
                pb = pr == c
                gb = gt == c
                inter = np.logical_and(pb, gb).sum()
                uni = np.logical_or(pb, gb).sum()
                intersect[c] += inter
                union[c] += uni
        ious = intersect / np.maximum(union, 1)
        valid = union > 0
        miou = float(ious[valid].mean()) if valid.any() else 0.0
        return {"mIoU": miou}


class PanopticMetricRunner(TaskMetricRunner):
    """MaskFormer (MMDet) — metric: PQ."""

    metric_name = "PQ"

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_detector = _require_mmdet()
        self.model = init_detector(config_path, checkpoint_path, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        assert self.model is not None
        # MaskFormer typically uses backbone features F1..F4 directly
        backbone = self.model.backbone
        feats = swin_feats_from_h(backbone, h)
        H, W = int(img_meta["height"]), int(img_meta["width"])
        try:
            from mmdet.structures import DetDataSample

            data_sample = DetDataSample()
            data_sample.set_metainfo(
                dict(img_shape=(H, W), ori_shape=(H, W), pad_shape=(H, W), img_id=img_meta["image_id"])
            )
            # panoptic head path differs by version; store feats for custom head call
            if hasattr(self.model, "panoptic_head"):
                results = self.model.panoptic_head.predict(feats, [data_sample], rescale=True)
                return {"image_id": img_meta["image_id"], "panoptic": results[0]}
            if hasattr(self.model, "simple_test"):
                # older API expects image tensor; not ideal for h-injection
                return {"image_id": img_meta["image_id"], "feats_only": True, "error": "need panoptic_head"}
        except Exception as e:
            return {"image_id": img_meta["image_id"], "error": str(e)}
        return {"image_id": img_meta["image_id"], "error": "unsupported MaskFormer API"}

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        # Full PQ needs panopticapi; keep a clear placeholder result if preds incomplete
        try:
            from panopticapi.evaluation import pq_compute
        except ImportError:
            return {
                "PQ": float("nan"),
                "note": "Install panopticapi and provide GT panoptic folder to compute PQ",
            }
        gt_folder = kwargs.get("gt_folder")
        pred_folder = kwargs.get("pred_folder")
        if not gt_folder or not pred_folder:
            return {"PQ": float("nan"), "note": "Need gt_folder and pred_folder for pq_compute"}
        results = pq_compute(ann_file, kwargs.get("pred_json"), gt_folder, pred_folder)
        return {"PQ": float(results["All"]["pq"])}


class PoseMetricRunner(TaskMetricRunner):
    """HigherHRNet (MMPose, original HRNet backbone) — metric: mAP-OKS."""

    metric_name = "mAP-OKS"

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_model = _require_mmpose()
        self.model = init_model(config_path, checkpoint_path, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Inject h as early HRNet feature when possible; else return error guidance.

        HigherHRNet uses HRNet (not Swin). Codec `out_channels` should match stem width
        (default 32). Full keypoint head wiring depends on mmpose version.
        """
        assert self.model is not None
        try:
            # Best-effort: if backbone has stage transitions, set first stream feature to h
            backbone = self.model.backbone if hasattr(self.model, "backbone") else self.model
            # Many mmpose models expect full image; document limitation
            if hasattr(self.model, "predict"):
                # Without image path, we only support feature injection hooks if present
                return {
                    "image_id": img_meta["image_id"],
                    "error": (
                        "HigherHRNet-from-h requires a project-specific backbone hook; "
                        "set pose.eval_from_image=true in config to run image-based fallback "
                        "after optional RGB decode, or implement HRNet stem replacement."
                    ),
                }
        except Exception as e:
            return {"image_id": img_meta["image_id"], "error": str(e)}
        return {"image_id": img_meta["image_id"], "error": "pose from-h not hooked"}

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        # Standard COCO keypoint eval when predictions are in COCO format
        valid = [p for p in predictions if p and "keypoints" in p]
        if not valid:
            return {
                "mAP-OKS": float("nan"),
                "note": "No keypoint predictions; implement HigherHRNet-from-h or provide COCO-format preds",
            }
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(valid)
        ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return {"mAP-OKS": float(ev.stats[0])}


def build_metric_runner(task: str, device: str = "cuda") -> TaskMetricRunner:
    task = task.lower()
    if task in ("detection", "det", "object_detection"):
        return DetectionMetricRunner(device=device, with_mask=False)
    if task in ("instance", "instance_seg", "instance_segmentation"):
        return DetectionMetricRunner(device=device, with_mask=True)
    if task in ("semantic", "semantic_seg", "semantic_segmentation"):
        return SemanticMetricRunner(device=device)
    if task in ("panoptic", "panoptic_seg", "panoptic_segmentation"):
        return PanopticMetricRunner(device=device)
    if task in ("pose", "pose_estimation"):
        return PoseMetricRunner(device=device)
    raise ValueError(f"Unknown task for metric runner: {task}")


# Suggested OpenMMLab config names (user must download matching weights)
DEFAULT_TASK_NET_CONFIGS = {
    # Official Swin-Transformer-Object-Detection zoo (same Cascade Mask R-CNN + Swin-B)
    "detection": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",  # mAP-bbox
    "instance": "configs/task_networks/cascade_mask_rcnn_swin_base_coco.py",  # mAP-mask
    "semantic": "configs/task_networks/upernet_swin-b_coco.py",
    "panoptic": "configs/task_networks/maskformer_swin-b_coco.py",
    "pose": "configs/task_networks/higherhrnet_w32_coco_wholebody.py",
}

DEFAULT_TASK_NET_CKPTS = {
    "detection": "checkpoints/task_networks/detection/model.pth",
    "instance": "checkpoints/task_networks/instance/model.pth",
    "semantic": "checkpoints/task_networks/semantic/model.pth",
    "panoptic": "checkpoints/task_networks/panoptic/model.pth",
    "pose": "checkpoints/task_networks/pose/model.pth",
}
