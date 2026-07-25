# Task-network configs (for metric evaluation)

Official detection / instance weights come from
[Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

Detection and instance segmentation share the same **Cascade Mask R-CNN + Swin-B**
checkpoint; metrics differ by head output (`bbox` vs `mask`).

| Task | Model | Official source |
|------|-------|-----------------|
| detection | **Cascade Mask R-CNN + Swin-B** | [config](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) / [ckpt](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth) |
| instance | **Cascade Mask R-CNN + Swin-B** (same) | same config / checkpoint as detection |
| semantic | UPerNet + Swin-B | MMSegmentation UPerNet Swin-B |
| panoptic | MaskFormer + Swin-B | MMDetection MaskFormer Swin-B |
| pose | HigherHRNet-W32 | MMPose HigherHRNet COCO-WholeBody (HRNet backbone) |

### Download detection / instance checkpoints

```bash
mkdir -p checkpoints/task_networks/detection checkpoints/task_networks/instance

# Cascade Mask R-CNN + Swin-B (shared by detection bbox + instance mask)
CKPT_URL=https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth
curl -L -o checkpoints/task_networks/detection/model.pth "$CKPT_URL"
cp checkpoints/task_networks/detection/model.pth checkpoints/task_networks/instance/model.pth
rm -f checkpoints/task_networks/detection/PLACEHOLDER checkpoints/task_networks/instance/PLACEHOLDER
```

Symlink or copy the official config over the stub:

```bash
ln -sf /path/to/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
  configs/task_networks/cascade_mask_rcnn_swin_base_coco.py
```

Eval YAML fields:

```yaml
# detection (mAP-bbox) and instance (mAP-mask) share the same config/weights
task_config: "./configs/task_networks/cascade_mask_rcnn_swin_base_coco.py"
task_checkpoint: "./checkpoints/task_networks/detection/model.pth"  # or .../instance/model.pth
```

**Channel note:** Swin-B F1 has 128 channels; both detection and instance TAIC use `out_channels: 128`.
