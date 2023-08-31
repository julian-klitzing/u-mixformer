_base_ = ['./umixformer_mit-b0-multi_8xb2-160k_ade20k-512x512.py']

# model settings
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64, num_heads=[1, 2, 5, 8], 
        num_layers=[3, 8, 27, 3],
        encoder_params=dict(interval=6)
        ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        decoder_params=dict(embed_dim=768,
                            num_heads=[8, 5, 2, 1],
                            pool_ratio=[1, 2, 4, 8],        
                            num_multi = 4),
        )
)