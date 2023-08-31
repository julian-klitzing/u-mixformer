_base_ = ['./umixformer_mscan_t_8xb1-160k_cityscapes-1024x1024.py']
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    decode_head=dict(
        feature_strides=[4, 8, 16, 32],
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(embed_dim=128,
                            num_heads=[8, 5, 2, 1],
                            pool_ratio=[1, 2, 4, 8]),
        num_classes=19
        )
    )