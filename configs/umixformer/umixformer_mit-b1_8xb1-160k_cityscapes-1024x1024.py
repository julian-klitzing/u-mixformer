_base_ = ['./umixformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# model settings
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512]
        )
)