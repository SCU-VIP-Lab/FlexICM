# STUB / pointer config for Cascade Mask R-CNN + Swin-B (official Swin detection zoo).
#
# Source repo: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
# Official config:
#   configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
# Official checkpoint:
#   https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth
#
# Replace this file by copying/symlinking the real config from that repo, then set
# task_checkpoint to the downloaded .pth under checkpoints/task_networks/detection/.
#
# Used for FlexICM detection (mAP-bbox) and instance segmentation (mAP-mask).
# Backbone F1 channels = 128 (Swin-B).

raise RuntimeError(
    "Replace configs/task_networks/cascade_mask_rcnn_swin_base_coco.py with the official "
    "Swin-Transformer-Object-Detection config: "
    "cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
)
