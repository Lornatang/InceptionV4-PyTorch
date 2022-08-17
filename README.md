# InceptionV3-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v3.pdf).

## Table of contents

- [InceptionV3-PyTorch](#inceptionv3-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Rethinking the Inception Architecture for Computer Vision](#rethinking-the-inception-architecture-for-computer-vision)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `inception_v3`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `test`.
- line 88: `model_weights_path` change to `./results/pretrained_models/InceptionV3-ImageNet_1K-b65ce284.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `inception_v3`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 47: `pretrained_model_weights_path` change to `./results/pretrained_models/InceptionV3-ImageNet_1K-b65ce284.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `inception_v3`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 50: `resume` change to `./samples/inception_v3-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1512.00567v3.pdf](https://arxiv.org/pdf/1512.00567v3.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|    Model     |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------:|:-----------:|:-----------------:|:-----------------:|
| inception_v3 | ImageNet_1K | 21.2%(**22.7%**)  |  5.6%(**6.5%**)   |

```bash
# Download `InceptionV3-ImageNet_1K-32d70693.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `inception_v3` model successfully.
Load `inception_v3` model weights `/InceptionV3-PyTorch/results/pretrained_models/InceptionV3-ImageNet_1K-32d70693.pth.tar` successfully.
tench, Tinca tinca                                                          (90.35%)
barracouta, snoek                                                           (1.21%)
rock beauty, Holocanthus tricolor                                           (0.06%)
armadillo                                                                   (0.04%)
electric ray, crampfish, numbfish, torpedo                                  (0.04%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Rethinking the Inception Architecture for Computer Vision

*Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, Zbigniew Wojna*

##### Abstract

Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks.
Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various
benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most
tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are
still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we are exploring ways to
scale up networks in ways that aim at utilizing the added computation as efficiently as possible. We benchmark our
methods on the ILSVRC 2012 classification challenge validation set and demonstrate substantial gains over the state of
the art via to carefully factorized convolutions and aggressive regularization: 21.2% top-1 and 5.6% top-5 error for
single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with
using less than 25 million parameters.

[[Paper]](https://arxiv.org/pdf/1409.4842v1.pdf)

```bibtex
@inproceedings{szegedy2016rethinking,
  title={Rethinking the inception architecture for computer vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2818--2826},
  year={2016}
}
```