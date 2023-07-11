_base_ = [
    '../../_base_/models/segformer_mit-b0.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]
load_from = "checkpoints/segmentation/feedformer/ade20k/B0/feedformer_b0_ade20k_030723.pth" # trained by Seul-Ki
randomness = dict(seed=0)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='checkpoints/classification/mit_b0.pth',
    backbone=dict(type="MixVisionTransformer_btn2"),
    decode_head=dict(
        type='FeedFormerHead',
        feature_strides=[4, 8, 16, 32],
        # in_channels=[32, 64, 160, 256],
        # in_index=[0, 1, 2, 3],
        # channels=128,
        num_classes=150
    )
)

# Use optimizer setting recommended by Andrej Karpathy as default
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=3e-4, weight_decay=0),       
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# ------ Leave learning rate constant during training ----------
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]

# Switch off the random augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
