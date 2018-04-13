# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np

from .basic_layers import ResidualBlockBottleNeck, ResidualBlockWide
from .attention_module import AttentionModule52_1, AttentionModule52_2, AttentionModule52_3, AttentionModule52_4


class ResidualAttentionNetwork(object):
    """
    Residual Attention Network for cifar-10.
    If you would like to use anther dataset, please override this class
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        self.input_shape = [-1, 32, 32, 3]
        self.output_dim = 10

        # for cifar-10, you should use attention module 2 for first stage
        self.attention_module_1 = AttentionModule52_2(scope="attention_module_1")
        # self.attention_module_1 = AttentionModule52_1(scope="attention_module_1")
        self.attention_module_2 = AttentionModule52_3(scope="attention_module_2")
        self.attention_module_3 = AttentionModule52_4(scope="attention_module_3")
        self.residual_block = ResidualBlockBottleNeck()

    def f_prop(self, x, is_training=True):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        # conv, x -> [None, row, line, 32]
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='SAME')

        # max pooling, x -> [None, row, line, 32]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # attention module, x -> [None, row, line, 16: when using widenet, filters=16*widen]
        x = self.attention_module_1.f_prop(x, filters=16, is_training=is_training)

        # residual block, x-> [None, row, line, 16: when using widenet, filters=16*widen]
        x = self.residual_block.f_prop(x, filters=32, scope="residual_block_1",
                                       is_training=is_training)
        # max pooling, x -> [None, row/2, line/2, 16]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # attention module, x -> [None, row/2, line/2, 32: when using widenet, filters=32*widen]
        x = self.attention_module_2.f_prop(x, filters=32, is_training=is_training)

        # residual block, x-> [None, row/2, line/2, when using widenet, filters=32*widen]
        x = self.residual_block.f_prop(x, filters=64, scope="residual_block_2",
                                       is_training=is_training)
        # max pooling, x -> [None, row/4, line/4, 128]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # attention module, x -> [None, row/4, line/4, 128]
        x = self.attention_module_3.f_prop(x, filters=64, is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, filters=128, scope="residual_block_3",
                                       is_training=is_training)
        # max pooling, x -> [None, row/4, line/4, 512]
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # residual block, x-> [None, row/4, line/4, 512]
        x = self.residual_block.f_prop(x, filters=192, scope="residual_block_4",
                                       is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 512]
        x = self.residual_block.f_prop(x, filters=256, scope="residual_block_5",
                                       is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 512]
        x = self.residual_block.f_prop(x, filters=512, scope="residual_block_6",
                                       is_training=is_training)

        # average pooling
        x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)

        return y

