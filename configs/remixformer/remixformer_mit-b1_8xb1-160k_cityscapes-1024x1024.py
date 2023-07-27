_base_ = ['./remixformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# model settings
model = dict(
    pretrained='checkpoints/classification/mit_b1.pth',
    backbone=dict(
        embed_dims=64),
    decode_head=dict(
        in_channels=[64, 128, 320, 512]
        )
)