# ReMixFormer

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch works with PyTorch 1.6+.

### 🎉 Introducing MMSegmentation v1.0.0 🎉

We are thrilled to announce the official release of MMSegmentation's latest version! For this new release, the [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch serves as the primary branch, while the development branch is [dev-1.x](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x). The stable branch for the previous release remains as the [0.x](https://github.com/open-mmlab/mmsegmentation/tree/0.x) branch. Please note that the [master](https://github.com/open-mmlab/mmsegmentation/tree/master) branch will only be maintained for a limited time before being removed. We encourage you to be mindful of branch selection and updates during use. Thank you for your unwavering support and enthusiasm, and let's work together to make MMSegmentation even more robust and powerful! 💪

MMSegmentation v1.x brings remarkable improvements over the 0.x release, offering a more flexible and feature-packed experience. To utilize the new features in v1.x, we kindly invite you to consult our detailed [📚 migration guide](https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html), which will help you seamlessly transition your projects. Your support is invaluable, and we eagerly await your feedback!

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## What's New

v1.0.0 was released on 04/06/2023.
Please refer to [changelog.md](docs/en/notes/changelog.md) for details and release history.

- Add Mapillary Vistas Datasets support to MMSegmentation Core Package ([#2576](https://github.com/open-mmlab/mmsegmentation/pull/2576))
- Support PIDNet ([#2609](https://github.com/open-mmlab/mmsegmentation/pull/2609))
- Support SegNeXt ([#2654](https://github.com/open-mmlab/mmsegmentation/pull/2654))

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb) on Colab.

To migrate from MMSegmentation 0.x, please refer to [migration](docs/en/migration).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<details open>
<summary>Supported backbones:</summary>

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)
- [x] [PoolFormer (CVPR'2022)](configs/poolformer)
- [x] [SegNeXt (NeurIPS'2022)](configs/segnext)

</details>

<details open>
<summary>Supported methods:</summary>

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)
- [x] [MaskFormer (NeurIPS'2021)](configs/maskformer)
- [x] [Mask2Former (CVPR'2022)](configs/mask2former)
- [x] [PIDNet (ArXiv'2022)](configs/pidnet)

</details>

<details open>
<summary>Supported datasets:</summary>

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid)
- [x] [Mapillary Vistas](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#mapillary-vistas-datasets)

</details>

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Projects

[Here](projects/README.md) are some implementations of SOTA models and solutions built on MMSegmentation, which are supported and maintained by community users. These projects demonstrate the best practices based on MMSegmentation for research and product development. We welcome and appreciate all the contributions to OpenMMLab ecosystem.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## OpenMMLab Family

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
