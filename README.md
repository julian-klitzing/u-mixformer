# U-MixFormer

## Introduction
<!-- 
### U-MixFormer

![demo image](resources/seg_demo.gif) -->

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Training

```
# Single-gpu training
python tools/train.py configs/remixformer/remixformer_mit-b0_8xb2-160k_ade20k-512x512.py

# Multi-gpu training
./tools/dist_train.sh configs/remixformer/remixformer_mit-b0_8xb2-160k_ade20k-512x512.py <GPU_NUM>
```

## Evaluation

```
# Single-gpu training
python tools/test.py configs/remixformer/remixformer_mit-b0_8xb2-160k_ade20k-512x512.py /path/to/checkpoint_file

# Multi-gpu training
./tools/dist_test.sh configs/remixformer/remixformer_mit-b0_8xb2-160k_ade20k-512x512.py /path/to/checkpoint_file <GPU_NUM>
```

## Qualitative Test (i.e. visualization)
### Visualization
```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out-file ${OUTPUT_IMAGE_NAME}] [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Example: visualize ```ReMixFormer-B0``` on ```cityscapes```: 

```shell
python demo/image_demo.py demo/demo.png configs/remixformer/remixformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py \
/path/to/checkpoint_file --out-file demo/output.png --device cuda:0 --palette cityscapes
```
### Zoom in the specific area (only for paper)
```shell
python paper/zoom_demo.py
```

### Make Figure No.1
Generate a SVG file
```shell
python paper/figure1.py
```

## Onnx Model Conversion
Please first install mmdeploy in another folder and run on mmsegmentation folder
```shell
python /path/to/MMDEPLOY_PATH/tools/deploy.py ${DEPLOY_CONFIG_FILE} ${MODEL_CONFIG} ${CHECKPOINT_FILE} ${IMAGE_FILE} \
[--work-dir ${SAVE_FOLDER_NAME}] [--device ${DEVICE_NAME}] [--dump-info]
```

Example: Deploy ```ReMixFormer-B0``` on ADE20K into ONNX model: 

```shell
python /path/to/MMDEPLOY_PATH/tools/deploy.py ../mmdeploy/configs/mmseg/segmentation_onnxruntime_static-512x512.py \
configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py checkpoints/segmentation/segformer/segformer_mit-b0_512x512_160k_ade20k.pth \
demo/demo.png \
--work-dir mmdeploy_model/segformer_mit_b0_ade_512x512 \
--device cuda \
--dump-info
```

## Table


## Citation

<!-- If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
``` -->

## License

This project is released under the [Apache 2.0 license](LICENSE).
