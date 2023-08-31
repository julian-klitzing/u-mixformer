_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_adamw.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
randomness = dict(seed=0) #seed setup
find_unused_parameters = True #find it in mmcv
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='APFormerHeadCity',  #FeedFormerHeadUNet, FeedFormerHeadUNetPlus, FeedFormerHead32, FeedFormerHead32_new'
        feature_strides=[4, 8, 16, 32],
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(embed_dim=128,
                            num_heads=[8, 5, 2, 1],
                            pool_ratio=[1, 2, 4, 8]),
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# val_evaluator = dict(type='SingleIoUMetric', iou_metrics=['mIoU']) #, output_dir='paper/mIoU')
# test_evaluator = val_evaluator

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
