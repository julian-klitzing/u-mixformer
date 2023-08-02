# ReMixFormer

## Introduction

### 🎉 ReMixFormer 🎉

![demo image](resources/seg_demo.gif)

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


## Visualization

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
