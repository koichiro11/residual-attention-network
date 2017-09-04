# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np

from .basic_layers import Dense, Conv, ResidualBlock
from .attention_module import AttentionModule


class ResidualAttentionModel(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        """
        :param input_shape: the list of input shape (ex: [None, 28, 28 ,3]
        :param output_dim:
        """

        self.average_pooling_kernel = None

    def __call__(self, target="CIFAR-10"):
        self.target = target
        self._build_model()

    def _build_model(self):
        """
        build model for specific data
        """
        if self.target == "ImageNet":
            """
            ImageNet.shape = [None, 224, 224, 3]
            """
            self.input_shape = [-1, 224, 224, 3]
            self.output_dim = 1000

            # conv, x -> [None, row/2, line/2, 64]
            self.conv1 = Conv([7, 7, self.input_shape[3], 64], strides=[1, 2, 2, 1])
            # max pooling, x -> [None, row/4, line/4, 64]

            # residual block, x -> [None, row/4, line/4, 128]
            self.residual_block1 = ResidualBlock(64, 128)
            # attention module, x -> [None, row/4, line/4, 128]
            self.attention_module1 = AttentionModule(128)
            # residual block, x -> [None, row/8, line/8, 256]
            self.residual_block2 = ResidualBlock(128, output_channels=256, stride=2)
            # attention module, x -> [None, row/8, line/8, 256]
            self.attention_module2 = AttentionModule(256)
            # residual block, x -> [None, row/16, line/16, 512]
            self.residual_block3 = ResidualBlock(256, output_channels=512, stride=2)
            # attention module, x -> [None, row/16, line/16, 512]
            self.attention_module3 = AttentionModule(512)
            # residual block, x -> [None, row/32, line/32, 1024]
            self.residual_block4 = ResidualBlock(512, output_channels=1024, stride=2)
            # FC, softmax, [None, 1024]
            self.average_pooling_kernel = [1, 7, 7, 1]
            self.dense = Dense([1024, self.output_dim])
        elif self.target == "CIFAR-10":
            """
            CIFER-10.shape = [None, 32, 32, 3]
            """
            self.input_shape = [-1, 32, 32, 3]
            self.output_dim = 10
            # conv, x -> [None, row, line, 64]
            self.conv1 = Conv([7, 7, self.input_shape[3], 32], strides=[1, 1, 1, 1])
            # max pooling, x -> [None, row/2, line/2, 64]

            # residual block, x -> [None, row/2, line/2, 128]
            self.residual_block1 = ResidualBlock(32, 64)
            # attention module, x -> [None, row/2, line/2, 128]
            self.attention_module1 = AttentionModule(64)
            # residual block, x -> [None, row/2, line/2, 256]
            # self.residual_block2 = ResidualBlock(128, output_channels=256, stride=1)
            # attention module, x -> [None, row/8, line/8, 256]
            # self.attention_module2 = AttentionModule(256)
            # residual block, x -> [None, row/4, line/4, 512]
            self.residual_block3 = ResidualBlock(64, output_channels=128, stride=2)
            # attention module, x -> [None, row/4, line/4, 512]
            self.attention_module3 = AttentionModule(512)
            # residual block, x -> [None, row/8, line/8, 1024]
            self.residual_block4 = ResidualBlock(128, output_channels=256, stride=2)
            # FC, softmax, [None, 1024]
            self.average_pooling_kernel = [1, 4, 4, 1]
            self.dense = Dense([256, self.output_dim])
        else:
            raise ValueError("this class is not for {target} dataset. Please write build_model method by yourself.".format(target=self.target))

    def f_prop(self, x):
        """
        forward propagation
        :param x: input x
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        # conv, x -> [None, row/2, line/2, 64]
        x = self.conv1.f_prop(x)

        # max pooling, x -> [None, row/4, line/4, 64]
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.residual_block1.f_prop(x)

        x = self.attention_module1.f_prop(x)

        # x = self.residual_block2.f_prop(x)

        # x = self.attention_module2.f_prop(x)

        x = self.residual_block3.f_prop(x)

        x = self.attention_module3.f_prop(x)

        x = self.residual_block4.f_prop(x)

        # average pooling
        x = tf.nn.avg_pool(x, ksize=self.average_pooling_kernel, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        # FC, softmax
        y = self.dense.f_prop(x)

        return y

