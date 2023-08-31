#!/usr/bin/env bash

PORT=29050 sh tools/dist_test.sh configs/umixformer/umixformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py checkpoints/segmentation/umixformer/cityscapes/umixformer.b0.1024.city.160k.pth 2

PORT=29030 sh tools/dist_test.sh configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py checkpoints/segmentation/segformer/segformer_mit-b0_1024x1024_160k_cityscapes.pth 2

PORT=29040 sh tools/dist_test.sh configs/feedformer/B0/feedformer.b0.1024x1024.city.160k.py checkpoints/segmentation/feedformer/city/B0/feedformer.b0.1024.city.160k.pth 2



