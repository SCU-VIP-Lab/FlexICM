"""Task-network metric runners: load official checkpoints and evaluate from codec feature h.

Paper flow (Sec.III.A):
  codec -> h (H/4 x W/4 x C) -> truncated task network (from Stage 2 / FPN) -> task output
  then compute mAP-bbox / mAP-mask / mIoU / PQ / mAP-OKS.

Requires optional packages:
  pip install pycocotools
  mim install mmdet mmsegmentation mmpose   # plus mmengine mmcv
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexicm.tasks.hrnet_features import hrnet_feats_from_h

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEMANTIC_EVAL_CLASSES_FILE = (
    _REPO_ROOT / "configs" / "eval" / "semantic_ade_coco_eval_classes.json"
)


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
        from pycocotools import mask as mask_util
        import numpy as np
        import torch

        def _encode_mask(mask) -> Dict[str, Any]:
            """Convert HxW bool/uint8 mask to COCO RLE."""
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask)
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            return rle

        coco_gt = COCO(ann_file)
        cat_ids = coco_gt.getCatIds()
        coco_results = []
        for pred in predictions:
            if pred is None:
                continue
            if "raw" in pred:
                # mmdet 2.x: (bbox_results, segm_results) or bbox_results only
                raw = pred["raw"]
                if isinstance(raw, tuple):
                    bbox_results, segm_results = raw[0], raw[1]
                else:
                    bbox_results, segm_results = raw, None
                for label, bboxes in enumerate(bbox_results):
                    segms = None
                    if self.with_mask and segm_results is not None:
                        segms = segm_results[label]
                    for i, row in enumerate(bboxes):
                        x1, y1, x2, y2, score = row[:5]
                        item = {
                            "image_id": int(pred["image_id"]),
                            "category_id": int(cat_ids[label])
                            if label < len(cat_ids)
                            else int(label + 1),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        }
                        if segms is not None and i < len(segms):
                            seg = segms[i]
                            if isinstance(seg, dict) and "counts" in seg:
                                # already RLE; ensure counts is str
                                counts = seg["counts"]
                                if isinstance(counts, bytes):
                                    seg = dict(seg)
                                    seg["counts"] = counts.decode("ascii")
                                item["segmentation"] = seg
                            else:
                                item["segmentation"] = _encode_mask(seg)
                        coco_results.append(item)
                continue

            bboxes = pred["bboxes"].numpy() if torch.is_tensor(pred["bboxes"]) else np.asarray(pred["bboxes"])
            scores = pred["scores"].numpy() if torch.is_tensor(pred["scores"]) else np.asarray(pred["scores"])
            labels = pred["labels"].numpy() if torch.is_tensor(pred["labels"]) else np.asarray(pred["labels"])
            masks = pred.get("masks", None)
            if masks is not None and torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()

            for i, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                lab = int(label)
                cat_id = int(cat_ids[lab]) if lab < len(cat_ids) else lab + 1
                item = {
                    "image_id": int(pred["image_id"]),
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
                if self.with_mask and masks is not None and i < len(masks):
                    item["segmentation"] = _encode_mask(masks[i])
                coco_results.append(item)

        if not coco_results:
            return {self.metric_name: 0.0}

        # BBox AP (always useful to log alongside mask)
        bbox_results = [{k: v for k, v in r.items() if k != "segmentation"} for r in coco_results]
        coco_dt_bbox = coco_gt.loadRes(bbox_results)
        coco_eval = COCOeval(coco_gt, coco_dt_bbox, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = {"mAP-bbox": float(coco_eval.stats[0])}

        if self.with_mask:
            n_with_seg = sum(1 for r in coco_results if "segmentation" in r)
            if n_with_seg == 0:
                metrics["mAP-mask"] = 0.0
                metrics["mAP-mask_error"] = "no segmentation fields in predictions"
            else:
                try:
                    coco_dt_seg = coco_gt.loadRes(coco_results)
                    coco_eval_m = COCOeval(coco_gt, coco_dt_seg, iouType="segm")
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
        dataset_meta = getattr(self.model, "dataset_meta", {}) or {}
        self.class_names = list(dataset_meta.get("classes") or [])

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

    @classmethod
    def _build_target_name_maps(cls, categories: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        exact_map: Dict[str, int] = {}
        root_buckets: Dict[str, List[int]] = {}
        for idx, category in enumerate(categories):
            raw_name = str(category.get("name", ""))
            exact_key = cls._normalize_class_name(raw_name)
            if exact_key and exact_key not in exact_map:
                exact_map[exact_key] = idx

            parts = [p for p in raw_name.strip().lower().split("-") if p]
            while parts and parts[-1] in {"merged", "other", "stuff"}:
                parts.pop()
            if not parts:
                continue
            root_key = cls._normalize_class_name("-".join(parts))
            if root_key:
                root_buckets.setdefault(root_key, []).append(idx)

        unique_root_map = {
            root: indices[0] for root, indices in root_buckets.items() if len(indices) == 1
        }
        return exact_map, unique_root_map

    @classmethod
    def _load_eval_class_allowlist(
        cls,
        eval_classes_file: Optional[str] = None,
        eval_class_names: Optional[Sequence[str]] = None,
    ) -> Optional[Set[str]]:
        """COCO class-name allowlist for partial ADE→COCO mIoU."""
        if eval_class_names:
            return {str(n).strip() for n in eval_class_names if str(n).strip()}
        path = eval_classes_file
        if not path:
            path = str(DEFAULT_SEMANTIC_EVAL_CLASSES_FILE)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            payload = json.load(f)
        names = payload.get("coco_classes") or payload.get("classes") or []
        return {str(n).strip() for n in names if str(n).strip()}

    @classmethod
    def _build_pred_to_gt_lookup(
        cls,
        pred_class_names: List[str],
        ann_file: str,
        ignore_label: int,
        eval_class_allowlist: Optional[Set[str]] = None,
    ):
        if not pred_class_names:
            return None, {}

        with open(ann_file) as f:
            coco = json.load(f)
        categories = coco.get("categories", [])
        exact_map, unique_root_map = cls._build_target_name_maps(categories)

        import numpy as np

        lookup = np.full(len(pred_class_names), ignore_label, dtype=np.int64)
        mapped_pairs = {}
        for src_idx, src_name in enumerate(pred_class_names):
            exact_key = cls._normalize_class_name(src_name)
            target_idx = exact_map.get(exact_key)
            if target_idx is None:
                raw_parts = [p for p in str(src_name).strip().lower().split("-") if p]
                while raw_parts and raw_parts[-1] in {"merged", "other", "stuff"}:
                    raw_parts.pop()
                root_key = cls._normalize_class_name("-".join(raw_parts))
                target_idx = unique_root_map.get(root_key)
            if target_idx is None:
                continue
            coco_name = str(categories[target_idx]["name"]).strip()
            if eval_class_allowlist is not None and coco_name not in eval_class_allowlist:
                continue
            lookup[src_idx] = target_idx
            mapped_pairs[str(src_name).strip()] = coco_name
        return lookup, mapped_pairs

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
        Paper-style ADE→COCO: only name-matched classes are scored; unmatched GT is ignore.
        """
        gt_loader = kwargs.get("gt_seg_loader")
        if gt_loader is None:
            return {
                "mIoU": float("nan"),
                "note": "Provide gt_seg_loader or use panoptic stuff GT to compute mIoU",
            }

        import numpy as np

        num_classes = int(kwargs.get("num_classes", 133))
        ignore_label = int(kwargs.get("ignore_label", 255))
        allowlist = self._load_eval_class_allowlist(
            eval_classes_file=kwargs.get("eval_classes_file"),
            eval_class_names=kwargs.get("eval_class_names"),
        )
        lookup, mapped_pairs = self._build_pred_to_gt_lookup(
            getattr(self, "class_names", []),
            ann_file,
            ignore_label=ignore_label,
            eval_class_allowlist=allowlist,
        )

        # ADE→COCO partial mIoU: name-matched classes, optionally filtered by allowlist.
        if lookup is not None and mapped_pairs:
            eval_classes = sorted(
                {int(x) for x in lookup.tolist() if int(x) != ignore_label}
            )
        else:
            eval_classes = list(range(num_classes))
        eval_class_arr = np.asarray(eval_classes, dtype=np.int64)
        eval_mask = np.zeros(num_classes, dtype=bool)
        if len(eval_class_arr):
            eval_mask[eval_class_arr] = True

        # Confusion with an extra sink column for ignore/out-of-range preds (counts as FN).
        hist = np.zeros((num_classes, num_classes + 1), dtype=np.float64)

        for pred in predictions:
            gt = gt_loader(pred["image_id"])
            pr = np.asarray(pred["seg"])
            if lookup is not None:
                valid = (pr >= 0) & (pr < len(lookup))
                remapped = np.full(pr.shape, ignore_label, dtype=np.int64)
                remapped[valid] = lookup[pr[valid]]
                pr = remapped
            if gt.shape != pr.shape:
                continue
            gt_eval = np.asarray(gt, dtype=np.int64)
            if lookup is not None and mapped_pairs:
                # Unmapped / OOB GT → ignore (excluded from FP/FN).
                in_range = (gt_eval >= 0) & (gt_eval < num_classes)
                mapped_ok = np.zeros(gt_eval.shape, dtype=bool)
                mapped_ok[in_range] = eval_mask[gt_eval[in_range]]
                gt_eval = np.where(mapped_ok, gt_eval, ignore_label)

            mask = (gt_eval != ignore_label) & (gt_eval >= 0) & (gt_eval < num_classes)
            if not np.any(mask):
                continue
            gt_m = gt_eval[mask]
            pr_m = pr[mask].astype(np.int64, copy=False)
            invalid = (pr_m < 0) | (pr_m >= num_classes) | (pr_m == ignore_label)
            pr_m = pr_m.copy()
            pr_m[invalid] = num_classes  # sink → FN only
            hist += np.bincount(
                gt_m * (num_classes + 1) + pr_m,
                minlength=num_classes * (num_classes + 1),
            ).reshape(num_classes, num_classes + 1)

        ious = []
        for c in eval_classes:
            tp = float(hist[c, c])
            # row includes sink FN; col is FP among non-ignore GT only
            union = float(hist[c, :].sum() + hist[:num_classes, c].sum() - tp)
            if union <= 0:
                continue
            ious.append(tp / union)
        miou = float(np.mean(ious)) if ious else 0.0
        result = {
            "mIoU": miou,
            "eval_classes": len(eval_classes),
            "eval_classes_with_gt": len(ious),
        }
        if allowlist is not None:
            result["eval_class_allowlist_size"] = len(allowlist)
        if lookup is not None:
            result["mapped_classes"] = len(mapped_pairs)
            if mapped_pairs:
                preview = ", ".join(f"{k}->{v}" for k, v in list(mapped_pairs.items())[:12])
                result["mapping_note"] = (
                    f"ADE→COCO partial mIoU over {len(eval_classes)} classes: {preview}"
                )
        return result


class PanopticMetricRunner(TaskMetricRunner):
    """MaskFormer / Mask2Former (MMDet) — metric: PQ.

    ``predict_from_h`` injects codec ``h`` as Swin F1, runs panoptic_head +
    panoptic_fusion_head, and returns a CPU panoptic map. ``finalize`` writes
    COCO-style RGB PNG + JSON under ``pred_folder`` (auto-created if needed)
    and calls ``panopticapi.evaluation.pq_compute``.

    """

    metric_name = "PQ"
    DEFAULT_EXCLUDE_STUFF = (
        "blanket",
        "pillow",
        "tent",
        "food-other-merged",
        "roof",
        "fruit",
        "paper-merged",
        "wall-stone",
        "banner",
    )

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_detector = _require_mmdet()
        self.model = init_detector(config_path, checkpoint_path, device=self.device)
        self.model.eval()
        self._label2cat = self._build_label2cat(ann_file=None)

    def _build_label2cat(self, ann_file: Optional[str] = None) -> Dict[int, int]:
        """Map contiguous model labels -> COCO category ids."""
        assert self.model is not None
        meta = getattr(self.model, "dataset_meta", None) or {}
        classes = [str(c).strip() for c in list(meta.get("classes") or [])]
        if not classes:
            return {}

        name_to_id: Dict[str, int] = {}
        if ann_file and os.path.isfile(ann_file):
            with open(ann_file) as f:
                coco = json.load(f)
            for cat in coco.get("categories", []):
                name_to_id[str(cat["name"]).strip()] = int(cat["id"])
        if not name_to_id:
            # Fallback when GT json is unavailable at load time.
            return {i: i + 1 for i in range(len(classes))}

        label2cat: Dict[int, int] = {}
        missing = []
        for i, name in enumerate(classes):
            if name in name_to_id:
                label2cat[i] = name_to_id[name]
            else:
                missing.append((i, name))
        if missing:
            print(
                f"[metric] panoptic label2cat missing {len(missing)} class name(s), "
                f"e.g. {missing[:5]} (will treat as VOID)"
            )
        return label2cat

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        assert self.model is not None
        backbone = self.model.backbone
        feats = swin_feats_from_h(backbone, h)

        ori_h, ori_w = int(img_meta["height"]), int(img_meta["width"])
        pad_h = int(img_meta.get("pad_height", h.shape[-2] * 4))
        pad_w = int(img_meta.get("pad_width", h.shape[-1] * 4))
        # Unpadded resized tensor size (codec pads after this).
        img_h = int(img_meta.get("img_height", pad_h))
        img_w = int(img_meta.get("img_width", pad_w))

        file_name = img_meta.get("file_name") or f"{int(img_meta['image_id']):012d}.jpg"
        segm_file = os.path.splitext(os.path.basename(str(file_name)))[0] + ".png"

        try:
            from mmdet.structures import DetDataSample

            if not hasattr(self.model, "panoptic_head") or not hasattr(
                self.model, "panoptic_fusion_head"
            ):
                return {
                    "image_id": img_meta["image_id"],
                    "file_name": segm_file,
                    "error": "model missing panoptic_head / panoptic_fusion_head",
                }

            data_sample = DetDataSample()
            data_sample.set_metainfo(
                dict(
                    img_shape=(img_h, img_w),
                    ori_shape=(ori_h, ori_w),
                    pad_shape=(pad_h, pad_w),
                    batch_input_shape=(pad_h, pad_w),
                    img_id=img_meta["image_id"],
                )
            )
            mask_cls, mask_pred = self.model.panoptic_head.predict(
                feats, [data_sample]
            )
            results_list = self.model.panoptic_fusion_head.predict(
                mask_cls, mask_pred, [data_sample], rescale=True
            )
            pan_results = results_list[0].get("pan_results")
            if pan_results is None:
                return {
                    "image_id": img_meta["image_id"],
                    "file_name": segm_file,
                    "error": "fusion head returned no pan_results",
                }
            sem = pan_results.sem_seg
            if hasattr(sem, "detach"):
                pan = sem.detach().cpu().numpy()
            else:
                pan = sem
            if pan.ndim == 3:
                pan = pan[0]
            return {
                "image_id": int(img_meta["image_id"]),
                "file_name": segm_file,
                "panoptic_seg": pan.astype("int64", copy=False),
            }
        except Exception as e:
            return {
                "image_id": img_meta["image_id"],
                "file_name": segm_file,
                "error": str(e),
            }

    def _export_predictions(
        self,
        predictions: List[Any],
        ann_file: str,
        pred_folder: str,
        pred_json: str,
    ) -> Tuple[str, str, int]:
        """Write RGB panoptic PNGs + JSON; return paths and #exported images."""
        import numpy as np
        from PIL import Image
        from mmdet.evaluation.functional import INSTANCE_OFFSET
        from panopticapi.utils import id2rgb

        os.makedirs(pred_folder, exist_ok=True)
        label2cat = self._build_label2cat(ann_file=ann_file) or self._label2cat
        meta = getattr(self.model, "dataset_meta", None) or {}
        num_classes = len(list(meta.get("classes") or []))
        ignore_index = int(meta.get("ignore_index", num_classes)) if num_classes else 255
        VOID = 0

        annotations = []
        n_err = 0
        for pred in predictions:
            if not pred:
                continue
            if pred.get("error") or "panoptic_seg" not in pred:
                n_err += 1
                continue
            pan = np.asarray(pred["panoptic_seg"]).copy()
            image_id = int(pred["image_id"])
            segm_file = pred.get("file_name") or f"{image_id:012d}.png"

            segments_info = []
            keep_ids = set()
            for pan_label in np.unique(pan):
                pan_label = int(pan_label)
                sem_label = pan_label % INSTANCE_OFFSET
                if sem_label == num_classes or sem_label == ignore_index:
                    continue
                if label2cat and sem_label not in label2cat:
                    continue
                mask = pan == pan_label
                area = int(mask.sum())
                if area <= 0:
                    continue
                cat_id = int(label2cat[sem_label]) if label2cat else int(sem_label)
                segments_info.append(
                    {
                        "id": pan_label,
                        "category_id": cat_id,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                keep_ids.add(pan_label)

            # panopticapi requires PNG ids ⊆ JSON segments_info (VOID=0 allowed).
            # Drop any leftover label (unmapped class / ignore) to VOID.
            if keep_ids:
                keep_mask = np.isin(pan, list(keep_ids))
                pan = np.where(keep_mask, pan, VOID).astype(pan.dtype, copy=False)
            else:
                pan = np.zeros_like(pan)

            rgb = id2rgb(pan.astype(np.uint32, copy=False)).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(pred_folder, segm_file))
            annotations.append(
                {
                    "image_id": image_id,
                    "file_name": segm_file,
                    "segments_info": segments_info,
                }
            )

        with open(ann_file) as f:
            gt = json.load(f)
        pred_payload = {
            "annotations": annotations,
            "categories": gt.get("categories", []),
            "images": gt.get("images", []),
        }
        os.makedirs(os.path.dirname(pred_json) or ".", exist_ok=True)
        with open(pred_json, "w") as f:
            json.dump(pred_payload, f)
        return pred_folder, pred_json, n_err

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        try:
            from panopticapi.evaluation import pq_compute
        except ImportError:
            return {
                "PQ": float("nan"),
                "note": "Install panopticapi: pip install git+https://github.com/cocodataset/panopticapi.git",
            }

        gt_folder = kwargs.get("gt_folder")
        if not gt_folder or not os.path.isdir(gt_folder):
            return {
                "PQ": float("nan"),
                "note": f"Need valid gt_folder for pq_compute (got {gt_folder!r})",
            }

        pred_folder = kwargs.get("pred_folder")
        pred_json = kwargs.get("pred_json")
        work_dir = kwargs.get("work_dir")
        if not pred_folder:
            base = work_dir or os.path.join(os.path.dirname(ann_file), "_flexicm_panoptic_pred")
            pred_folder = os.path.join(base, "panoptic_pred")
        if not pred_json:
            pred_json = os.path.join(
                os.path.dirname(pred_folder.rstrip(os.sep)) or pred_folder,
                "panoptic_pred.json",
            )

        # Always (re)export from in-memory predictions so PNG/JSON stay in sync.
        pred_folder, pred_json, n_err = self._export_predictions(
            predictions, ann_file, pred_folder, pred_json
        )
        print(f"[metric] wrote panoptic preds:\n  folder={pred_folder}\n  json={pred_json}")

        # pq_compute requires a prediction for every GT image; if we only ran a
        # subset (max_batches), evaluate against a filtered GT json.
        pred_ids = set()
        with open(pred_json) as f:
            pred_data = json.load(f)
        for ann in pred_data.get("annotations", []):
            pred_ids.add(int(ann["image_id"]))
        if not pred_ids:
            return {
                "PQ": float("nan"),
                "note": "No valid panoptic predictions to evaluate",
                "num_image_errors": n_err,
                "pred_folder": pred_folder,
                "pred_json": pred_json,
            }

        with open(ann_file) as f:
            gt_data = json.load(f)
        gt_ids = {int(a["image_id"]) for a in gt_data.get("annotations", [])}
        eval_ann_file = ann_file
        if pred_ids != gt_ids:
            filtered = dict(gt_data)
            filtered["annotations"] = [
                a for a in gt_data["annotations"] if int(a["image_id"]) in pred_ids
            ]
            filtered["images"] = [
                im for im in gt_data.get("images", []) if int(im["id"]) in pred_ids
            ]
            eval_ann_file = pred_json.replace(".json", "_gt_subset.json")
            with open(eval_ann_file, "w") as f:
                json.dump(filtered, f)
            print(
                f"[metric] subset PQ: {len(pred_ids)}/{len(gt_ids)} images "
                f"(filtered GT -> {eval_ann_file})"
            )

        results = pq_compute(eval_ann_file, pred_json, gt_folder, pred_folder)
        out = {
            "PQ": float(results["All"]["pq"]) * 100.0,  # report as percent
            "SQ": float(results["All"]["sq"]) * 100.0,
            "RQ": float(results["All"]["rq"]) * 100.0,
            "PQ_th": float(results["Things"]["pq"]) * 100.0,
            "PQ_st": float(results["Stuff"]["pq"]) * 100.0,
            "num_images": len(pred_ids),
            "pred_folder": pred_folder,
            "pred_json": pred_json,
            "num_stuff_classes": int(results["Stuff"]["n"]),
            "num_things_classes": int(results["Things"]["n"]),
        }

        if "panoptic_exclude_stuff" in kwargs:
            exclude = kwargs.get("panoptic_exclude_stuff")
        elif "exclude_stuff" in kwargs:
            exclude = kwargs.get("exclude_stuff")
        else:
            exclude = list(self.DEFAULT_EXCLUDE_STUFF)
        if exclude is None:
            exclude = list(self.DEFAULT_EXCLUDE_STUFF)
        if exclude:
            filtered = self._pq_macro_excluding(
                eval_ann_file, pred_json, gt_folder, pred_folder, exclude
            )
            out["PQ_full"] = out["PQ"]
            out["PQ_st_full"] = out["PQ_st"]
            out["PQ"] = filtered["All"]["pq"]
            out["SQ"] = filtered["All"]["sq"]
            out["RQ"] = filtered["All"]["rq"]
            out["PQ_th"] = filtered["Things"]["pq"]
            out["PQ_st"] = filtered["Stuff"]["pq"]
            out["num_stuff_classes"] = filtered["Stuff"]["n"]
            out["num_things_classes"] = filtered["Things"]["n"]
            out["panoptic_exclude_stuff"] = filtered["excluded"]
            print(
                f"[metric] PQ with stuff exclusions ({len(filtered['excluded'])} dropped): "
                f"All={out['PQ']:.2f}  Things={out['PQ_th']:.2f}  "
                f"Stuff={out['PQ_st']:.2f} (N_st={out['num_stuff_classes']})"
            )

        if n_err:
            out["num_image_errors"] = n_err
        return out

    @staticmethod
    def _pq_macro_excluding(
        gt_json: str,
        pred_json: str,
        gt_folder: str,
        pred_folder: str,
        exclude: List[Any],
    ) -> Dict[str, Any]:
        """Recompute PQ/SQ/RQ macro averages after dropping stuff categories."""
        from panopticapi.evaluation import pq_compute_multi_core

        with open(gt_json) as f:
            gt_obj = json.load(f)
        with open(pred_json) as f:
            pred_obj = json.load(f)

        categories = {int(c["id"]): c for c in gt_obj["categories"]}
        name2id = {str(c["name"]).strip().lower(): int(c["id"]) for c in gt_obj["categories"]}
        exclude_ids = set()
        excluded_names = []
        for item in exclude:
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                cid = int(item)
            else:
                cid = name2id.get(str(item).strip().lower())
                if cid is None:
                    raise KeyError(f"Unknown panoptic_exclude_stuff category: {item!r}")
            if cid not in categories:
                raise KeyError(f"Unknown panoptic category id: {cid}")
            if int(categories[cid].get("isthing", 0)) == 1:
                raise ValueError(f"panoptic_exclude_stuff expects stuff only, got thing: {item}")
            exclude_ids.add(cid)
            excluded_names.append(categories[cid]["name"])

        pred_by_id = {int(a["image_id"]): a for a in pred_obj["annotations"]}
        matched = [
            (gt_ann, pred_by_id[int(gt_ann["image_id"])])
            for gt_ann in gt_obj["annotations"]
            if int(gt_ann["image_id"]) in pred_by_id
        ]
        pq_stat = pq_compute_multi_core(matched, gt_folder, pred_folder, categories)

        def _avg(isthing: Optional[bool] = None) -> Dict[str, float]:
            pq = sq = rq = 0.0
            n = 0
            for cid, info in categories.items():
                if cid in exclude_ids:
                    continue
                if isthing is not None and bool(info.get("isthing", 0)) != bool(isthing):
                    continue
                iou = pq_stat[cid].iou
                tp = pq_stat[cid].tp
                fp = pq_stat[cid].fp
                fn = pq_stat[cid].fn
                if tp + fp + fn == 0:
                    continue
                n += 1
                pq += iou / (tp + 0.5 * fp + 0.5 * fn)
                sq += (iou / tp) if tp != 0 else 0.0
                rq += tp / (tp + 0.5 * fp + 0.5 * fn)
            if n == 0:
                return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}
            return {
                "pq": 100.0 * pq / n,
                "sq": 100.0 * sq / n,
                "rq": 100.0 * rq / n,
                "n": n,
            }

        return {
            "All": _avg(None),
            "Things": _avg(True),
            "Stuff": _avg(False),
            "excluded": excluded_names,
        }



class PoseMetricRunner(TaskMetricRunner):
    """HigherHRNet / AE-HRNet (MMPose, original HRNet backbone) — metric: mAP-OKS."""

    metric_name = "mAP-OKS"

    def load(self, config_path: str, checkpoint_path: str) -> None:
        init_model = _require_mmpose()
        # Register AEHigherResolutionHead via config custom_imports.
        self.model = init_model(config_path, checkpoint_path, device=self.device)
        self.model.eval()

    def _prepare_image_tensor(self, image, h: torch.Tensor):
        if image is not None and not torch.is_tensor(image):
            image = None
        if image is not None and image.dim() == 3:
            image = image.unsqueeze(0)
        if image is not None:
            image = image.to(device=h.device, dtype=h.dtype)
        return image

    def _decode_instances_from_pred(
        self,
        inst,
        img_meta: Dict[str, Any],
        pad_h: int,
        pad_w: int,
        feat_h: int,
        feat_w: int,
    ) -> Dict[str, Any]:
        import numpy as np

        keypoints = getattr(inst, "keypoints", None)
        keypoint_scores = getattr(inst, "keypoint_scores", None)
        bbox_scores = getattr(inst, "bbox_scores", None)
        if keypoints is None:
            return {"image_id": img_meta["image_id"], "instances": []}

        sf = img_meta.get("scale_factor", (1.0, 1.0))
        if isinstance(sf, (int, float)):
            scale_w = scale_h = float(sf)
        else:
            scale_w, scale_h = float(sf[0]), float(sf[1])

        kpts = np.asarray(keypoints, dtype=np.float32)
        kpt_scores = (
            np.asarray(keypoint_scores, dtype=np.float32)
            if keypoint_scores is not None
            else np.ones(kpts.shape[:2], dtype=np.float32)
        )
        if bbox_scores is None:
            inst_scores = kpt_scores.mean(axis=-1)
        else:
            inst_scores = np.asarray(bbox_scores, dtype=np.float32).reshape(-1)

        # AE decode often returns coords on the backbone feature grid (stride 4).
        # Map feature-space -> padded canvas -> original COCO image coords.
        sx = float(pad_w) / float(max(feat_w, 1))
        sy = float(pad_h) / float(max(feat_h, 1))
        instances = []
        for i in range(kpts.shape[0]):
            xy = kpts[i].copy()  # (K, 2)
            # If already on canvas scale, sx/sy≈1; if on /4 feature grid, sx/sy≈4.
            max_x = float(np.nanmax(xy[:, 0])) if xy.size else 0.0
            max_y = float(np.nanmax(xy[:, 1])) if xy.size else 0.0
            if max_x <= feat_w * 1.5 and max_y <= feat_h * 1.5:
                xy[:, 0] = xy[:, 0] * sx
                xy[:, 1] = xy[:, 1] * sy
            xy[:, 0] = xy[:, 0] / max(scale_w, 1e-6)
            xy[:, 1] = xy[:, 1] / max(scale_h, 1e-6)
            score_i = float(inst_scores[i]) if i < len(inst_scores) else float(kpt_scores[i].mean())
            if score_i < 0.05:
                continue
            flat = []
            for j in range(xy.shape[0]):
                # COCO visibility flag: 0/1/2; use 2 when confidence is decent
                vis = 2.0 if float(kpt_scores[i, j]) >= 0.05 else 0.0
                flat.extend([float(xy[j, 0]), float(xy[j, 1]), vis])
            instances.append(
                {
                    "keypoints": flat,
                    "score": score_i,
                    "num_keypoints": int(xy.shape[0]),
                }
            )
        return {"image_id": img_meta["image_id"], "instances": instances}

    def _make_pose_datasample(self, pad_h: int, pad_w: int):
        import numpy as np
        from mmpose.structures import PoseDataSample

        meta = getattr(self.model, "dataset_meta", None) or {}
        flip_indices = meta.get("flip_indices", list(range(17)))
        ds = PoseDataSample()
        ds.set_metainfo(
            dict(
                img_shape=(pad_h, pad_w),
                ori_shape=(pad_h, pad_w),
                input_size=(pad_w, pad_h),
                input_center=np.array([pad_w / 2.0, pad_h / 2.0], dtype=np.float32),
                input_scale=np.array([float(pad_w), float(pad_h)], dtype=np.float32),
                flip_indices=flip_indices,
            )
        )
        return ds

    @torch.no_grad()
    def predict_from_h(self, h: torch.Tensor, img_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Inject codec ``h`` as HRNet stage2 branch-0, then AE head → COCO keypoints."""
        assert self.model is not None
        try:
            backbone = self.model.backbone
            image = self._prepare_image_tensor(img_meta.get("image_tensor", None), h)
            feats = hrnet_feats_from_h(backbone, h, image=image)

            pad_h = int(img_meta.get("pad_height", h.shape[-2] * 4))
            pad_w = int(img_meta.get("pad_width", h.shape[-1] * 4))
            ds = self._make_pose_datasample(pad_h, pad_w)
            test_cfg = dict(getattr(self.model, "test_cfg", {}) or {})
            # Flip-TTA needs a flipped image feature; keep single-scale from-h.
            test_cfg["flip_test"] = False
            test_cfg["multiscale_test"] = False
            preds = self.model.head.predict(feats, [ds], test_cfg=test_cfg)
            inst = preds[0] if not isinstance(preds, tuple) else preds[0][0]
            return self._decode_instances_from_pred(
                inst, img_meta, pad_h, pad_w, int(h.shape[-2]), int(h.shape[-1])
            )
        except Exception as e:
            return {"image_id": img_meta["image_id"], "error": str(e), "instances": []}

    @torch.no_grad()
    def predict_from_ms_h(
        self,
        ms_pack: Sequence[Dict[str, Any]],
        img_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Multi-scale from-h: each pack is one resolution through the codec.

        ``ms_pack[0]`` must be the base scale (``1.0``): tags and decode use its
        padded canvas; other scales contribute averaged heatmaps only.
        """
        assert self.model is not None
        if not ms_pack:
            return {"image_id": img_meta["image_id"], "instances": []}
        if len(ms_pack) == 1:
            pack = ms_pack[0]
            meta = dict(img_meta)
            meta["image_tensor"] = pack.get("image_tensor")
            meta["pad_height"] = pack["pad_height"]
            meta["pad_width"] = pack["pad_width"]
            return self.predict_from_h(pack["h"], meta)

        try:
            backbone = self.model.backbone
            feats_list = []
            for pack in ms_pack:
                h = pack["h"]
                image = self._prepare_image_tensor(pack.get("image_tensor"), h)
                feats = hrnet_feats_from_h(backbone, h, image=image)
                # AEHigherResolutionHead MS path expects List[Tuple[Tensor, ...]]
                feats_list.append(tuple(feats))

            base = ms_pack[0]
            pad_h = int(base["pad_height"])
            pad_w = int(base["pad_width"])
            h0 = base["h"]
            ds = self._make_pose_datasample(pad_h, pad_w)
            test_cfg = dict(getattr(self.model, "test_cfg", {}) or {})
            test_cfg["flip_test"] = False
            test_cfg["multiscale_test"] = True
            preds = self.model.head.predict(feats_list, [ds], test_cfg=test_cfg)
            inst = preds[0] if not isinstance(preds, tuple) else preds[0][0]
            return self._decode_instances_from_pred(
                inst, img_meta, pad_h, pad_w, int(h0.shape[-2]), int(h0.shape[-1])
            )
        except Exception as e:
            return {"image_id": img_meta["image_id"], "error": str(e), "instances": []}

    def finalize(self, predictions: List[Any], ann_file: str, **kwargs) -> Dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_results = []
        n_err = 0
        for pred in predictions:
            if not pred:
                continue
            if pred.get("error"):
                n_err += 1
            # Legacy single-instance format
            if "keypoints" in pred and "instances" not in pred:
                coco_results.append(
                    {
                        "image_id": int(pred["image_id"]),
                        "category_id": 1,
                        "keypoints": pred["keypoints"],
                        "score": float(pred.get("score", 1.0)),
                    }
                )
                continue
            for inst in pred.get("instances") or []:
                coco_results.append(
                    {
                        "image_id": int(pred["image_id"]),
                        "category_id": 1,
                        "keypoints": inst["keypoints"],
                        "score": float(inst.get("score", 1.0)),
                    }
                )

        if not coco_results:
            note = "No keypoint predictions"
            if n_err:
                note += f" ({n_err} images raised errors; see pred['error'])"
                # Surface a sample error so full-val NaNs are debuggable.
                for pred in predictions:
                    if pred and pred.get("error"):
                        note += f" e.g. {pred['error']}"
                        break
            return {"mAP-OKS": float("nan"), "note": note}

        import numpy as np

        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(coco_results)
        ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        ev.evaluate()
        ev.accumulate()
        precision = ev.eval["precision"]
        iou_thrs = np.asarray(ev.params.iouThrs, dtype=np.float64)
        idx = np.where((iou_thrs >= 0.50 - 1e-9) & (iou_thrs <= 0.80 + 1e-9))[0]
        s = precision[idx, :, :, 0, -1]
        s = s[s > -1]
        out = {
            "mAP-OKS": float(np.mean(s)) if s.size else float("nan"),
            "num_predictions": len(coco_results),
        }
        if n_err:
            out["num_image_errors"] = n_err
        return out


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
    "semantic": "checkpoints/task_networks/semantic/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py",
    "panoptic": "checkpoints/task_networks/panoptic/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py",
    "pose": "checkpoints/task_networks/pose/higherhrnet_w32_coco_512x512.py",
}

DEFAULT_TASK_NET_CKPTS = {
    "detection": "checkpoints/task_networks/detection/model_mmdet3.pth",
    "instance": "checkpoints/task_networks/instance/model_mmdet3.pth",
    "semantic": "checkpoints/task_networks/semantic/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth",
    "panoptic": "checkpoints/task_networks/panoptic/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth",
    "pose": "checkpoints/task_networks/pose/higher_hrnet32_coco_512x512-8ae85183_20200713_mmpose1.pth",
}
