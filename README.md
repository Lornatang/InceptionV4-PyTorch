# InceptionV4-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v2.pdf).

## Table of contents

- [InceptionV4-PyTorch](#inceptionv4-pytorch)
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
        - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](#inception-v4-inception-resnet-and-the-impact-of-residual-connections-on-learning)

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

- line 29: `model_arch_name` change to `inception_v4`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 91: `model_weights_path` change to `./results/pretrained_models/InceptionV4-ImageNet_1K-2069673f.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `inception_v4`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/InceptionV4-ImageNet_1K-2069673f.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `inception_v4`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/inception_v4-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1602.07261v2.pdf](https://arxiv.org/pdf/1602.07261v2.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|         Model          |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:----------------------:|:-----------:|:-----------------:|:-----------------:|
|      inception_v4      | ImageNet_1K | 20.0%(**22.2%**)  |  5.0%(**6.1%**)   |
| inception_v4-resnet_v2 | ImageNet_1K | 19.9%(**23.5%**)  |  4.9%(**6.6%**)   |

```bash
# Download `InceptionV4-ImageNet_1K-2069673f.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `inception_v4` model successfully.
Load `inception_v4` model weights `/InceptionV4-PyTorch/results/pretrained_models/InceptionV4-ImageNet_1K-2069673f.pth.tar` successfully.
tench, Tinca tinca                                                          (85.94%)
barracouta, snoek                                                           (1.43%)
reel                                                                        (0.18%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.16%)
goldfish, Carassius auratus                                                 (0.09%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

*Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi*

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