auto_scale_lr = dict(base_batch_size=192)
backend_args = dict(backend='local')
codec = dict(
    decode_center_shift=0.5,
    decode_keypoint_order=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        11,
        12,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
    ],
    decode_max_instances=30,
    decode_topk=30,
    heatmap_size=(
        128,
        128,
    ),
    input_size=(
        512,
        512,
    ),
    sigma=2,
    type='AssociativeEmbedding')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'bottomup'
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=50,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    32,
                    64,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    32,
                    64,
                    128,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    32,
                    64,
                    128,
                    256,
                ),
                num_modules=3)),
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
            type='Pretrained'),
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            decode_center_shift=0.5,
            decode_keypoint_order=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                11,
                12,
                7,
                8,
                9,
                10,
                13,
                14,
                15,
                16,
            ],
            decode_max_instances=30,
            decode_topk=30,
            heatmap_size=(
                512,
                512,
            ),
            input_size=(
                512,
                512,
            ),
            sigma=2,
            type='AssociativeEmbedding'),
        deconv_out_channels=None,
        in_channels=32,
        keypoint_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        num_keypoints=17,
        tag_dim=1,
        tag_loss=dict(loss_weight=0.001, type='AssociativeEmbeddingLoss'),
        tag_per_keypoint=True,
        type='AssociativeEmbeddingHead'),
    test_cfg=dict(
        align_corners=False,
        flip_test=True,
        multiscale_test=False,
        restore_heatmap_size=True,
        shift_heatmap=False),
    type='BottomupPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.0015, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=300,
        gamma=0.1,
        milestones=[
            200,
            260,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    512,
                    512,
                ),
                resize_mode='expand',
                size_factor=64,
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'skeleton_links',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file='annotations/person_keypoints_train2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        pipeline=[],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = []
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    512,
                    512,
                ),
                resize_mode='expand',
                size_factor=64,
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'skeleton_links',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        input_size=(
            512,
            512,
        ),
        resize_mode='expand',
        size_factor=64,
        type='BottomupResize'),
    dict(
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'skeleton_links',
        ),
        type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
