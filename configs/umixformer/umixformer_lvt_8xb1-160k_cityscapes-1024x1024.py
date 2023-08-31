_base_ = [
        '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_adamw.py'
]
checkpoint = './checkpoints/classification/lvt_imagenet_pretrained.pth.tar'  # noqa
randomness = dict(seed=0) #seed setup
find_unused_parameters = True #find it in mmcv
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='lvt',
        in_channels=3,
        num_layers=[2, 2, 2, 2],
        patch_sizes=4,
        embed_dims=[64, 64, 160, 256],
        num_heads=[2, 2, 5, 8],
        mlp_ratios=[4, 8, 4, 4],
        mlp_depconv=[False, True, True, True],
        sr_ratios=[8, 4, 2, 1],
        sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
        qkv_bias=False,
        with_cls_head = False, # classification/downstream tasks
        rasa_cfg = dict(
            atrous_rates= [1,3,5], # None, [1,3,5]
            act_layer= 'nn.SiLU(True)',
            init= 'kaiming',
            r_num = 2,
        ), # rasa setting
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        ),
    decode_head=dict(
        type='APFormerHeadCity',
        feature_strides=[4, 8, 16, 32],
        in_channels=[64, 64, 160, 256],
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
        type='PolyLR', #check CosineAnnealingLR: https://github.com/open-mmlab/mmengine/blob/04b0ffee76c41d10c5fd1f737cdc5306af365754/mmengine/optim/scheduler/lr_scheduler.py#L48
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader