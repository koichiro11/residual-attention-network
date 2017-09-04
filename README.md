# Residual Attention Network for Image Classification@tensorflow

## Information of paper

- author

```
Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
```

- submission date

```
Submitted on 23 Apr 2017
```

- society

```
accepted to CVPR2017
```

- arxiv

```
https://arxiv.org/abs/1704.06904
```

- abstract

```
In this work, we propose "Residual Attention Network", a convolutional neural network using attention mechanism
which can incorporate with state-of-art feed forward network architecture in an end-to-end training fashion.
Our Residual Attention Network is built by stacking Attention Modules which generate attention-aware features.
The attention-aware features from different modules change adaptively as layers going deeper.
Inside each Attention Module, bottom-up top-down feedforward structure is used to unfold the feedforward and feedback attention process into a single feedforward process.
Importantly, we propose attention residual learning to train very deep Residual Attention Networks which can be easily scaled up to hundreds of layers.
Extensive analyses are conducted on CIFAR-10 and CIFAR-100 datasets to verify the effectiveness of every module mentioned above.
Our Residual Attention Network achieves state-of-the-art object recognition performance on three benchmark datasets including CIFAR-10 (3.90% error),
CIFAR-100 (20.45% error) and ImageNet (4.8% single model and single crop, top-5 error).
Note that, our method achieves 0.6% top-1 accuracy improvement with 46% trunk depth and 69% forward FLOPs comparing to ResNet-200.
The experiment also demonstrates that our network is robust against noisy labels.
```

## About this code & explanation

### purpose

This code is written by [Koichiro Tamura](http://koichirotamura.com/)
in order to understand [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904).


### explanation

I made document which explains [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)(in Japanese).
The document would be on SlideShare and the website of [Deep Learning JP](http://deeplearning.jp/en/)


### how to use

#### Requirements

- Anaconda3.x
- tensorflow 1.x
- keras 2.x

#### train

```
$ python residual-attention-network/train.py
```

#### test

```
Sorry, I have not written scripts for test.
```

