_base_ = './vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='checkpoints/segmentation/deit/upernet_deit-s16_512x512_80k_ade20k_20210624_095228-afc93ec2.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(
        type='FeedFormerHead',
        feature_strides=[4, 8, 16, 32],
        # in_channels=[384, 384, 384, 384],
        # in_index=[0, 1, 2, 3],
        # channels=128,
        num_classes=150
        ),
    neck=None,
    auxiliary_head=dict(num_classes=150, in_channels=384))
