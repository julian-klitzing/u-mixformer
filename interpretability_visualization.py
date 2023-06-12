"""
This script mainly adopts the functionality of the image_demo.py script i.e.,
given a config file it does an inference step on a certain query image. 
Furthermore it extract the interpretability measures and saves them into a dedicated
folder.
"""

# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


import mmcv
import torch
# from mmseg.ops import resize
from torchinfo import summary



def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # model.show_result(args.img, result, out_file="result2.png")

    
# ------------ Try to use torch summary ------------    
    # model.forward = model.forward_dummy 
    model_summary = summary(model, (1, 3, 512, 683))
# ------------ Try to use torch summary ------------


# ------------ Vizualize rollout ------------
    mask = model.backbone.rollout()

# ------------ Vizualize rollout ------------      
    
    
    
# ------------ Vizualize individual stages ------------   
    stage_to_visualize = 0
    channel_to_visualize = 350

    h1, w1 = model.backbone.attention_maps[0].shape[2:]
    attention_matrix = model.backbone.attention_maps[stage_to_visualize]
    
    attention_matrix = resize(attention_matrix, size=(h1, w1), mode='bilinear', align_corners=False)
    # Average over attentin heads
    attention_matrix = attention_matrix.mean(0)
    # Select channel to vizualize
    attention_matrix_channel = attention_matrix[channel_to_visualize]
    # Normalize values
    attention_matrix_channel = (attention_matrix_channel - attention_matrix_channel.min()) / (attention_matrix_channel.max() - attention_matrix_channel.min())
    attention_matrix_channel = attention_matrix_channel.cpu().numpy() * 255
    imwrite(attention_matrix_channel, f'./visualization/attention_stage_{stage_to_visualize}_chan_{channel_to_visualize}.png')
    # ------------ Vizualize individual stages ------------   


    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()



