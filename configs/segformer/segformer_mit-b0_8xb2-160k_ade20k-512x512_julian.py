_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# Trained weights by authors (converted to new mmseg with provided tool), so weights without bottleneck
load_from = "checkpoints/segmentation/segformer/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth" # to initialize entire model with pretrained weights

# load_from = "work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/20230629_150333/iter_160000.pth" # 

randomness = dict(seed=0) #seed setup

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(type='MixVisionTransformer_btn'),
    # pretrained='checkpoints/classification/mit_b0.pth', # only initalizes backbone
    decode_head=dict(num_classes=150))

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
#         type='LinearLR', start_factor=1e-6  , by_epoch=False, begin=0, end=1500), # start_factor=1e-6
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]
param_scheduler = None

train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader