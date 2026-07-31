# Cascade Mask R-CNN + Swin-B for FlexICM metric eval (mmdet 3.x).
# Matches the official Swin-Transformer-Object-Detection paper checkpoint:
#   cascade_mask_rcnn_swin_base_patch4_window7.pth
# Use the converted weights:
#   checkpoints/task_networks/detection/model_mmdet3.pth
# (produced by scripts/convert_swin_det_ckpt_to_mmdet3.py)

_base_ = []  # self-contained for FlexICM

model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=False,
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# Minimal dataset meta + test pipeline required by mmdet.apis.inference_detector
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(
            classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                     'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')))
)
