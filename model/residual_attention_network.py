# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np

from .basic_layers import ResidualBlock
from .attention_module import AttentionModule


class ResidualAttentionNetwork(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        """
        :param input_shape: the list of input shape (ex: [None, 28, 28 ,3]
        :param output_dim:
        """
        self.input_shape = [-1, 32, 32, 3]
        self.output_dim = 10

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()

    def f_prop(self, x, is_training=True):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        # conv, x -> [None, row, line, 32]
        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')

        # max pooling, x -> [None, row, line, 32]
        x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')

        # attention module, x -> [None, row, line, 32]
        x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_1", is_training=is_training)

        # residual block, x-> [None, row, line, 64]
        x = self.residual_block.f_prop(x, input_channels=32, output_channels=64, scope="residual_block_1",
                                       is_training=is_training)
        # max pooling, x -> [None, row, line, 64]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # attention module, x -> [None, row, line, 64]
        x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_2", is_training=is_training)

        # residual block, x-> [None, row, line, 128]
        x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_block_2",
                                       is_training=is_training)
        # max pooling, x -> [None, row/2, line/2, 128]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # attention module, x -> [None, row/2, line/2, 64]
        x = self.attention_module.f_prop(x, input_channels=128, scope="attention_module_3", is_training=is_training)

        # residual block, x-> [None, row/2, line/2, 256]
        x = self.residual_block.f_prop(x, input_channels=128, output_channels=256, scope="residual_block_3",
                                       is_training=is_training)
        # max pooling, x -> [None, row/4, line/4, 256]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_4",
                                       is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_5",
                                       is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_6",
                                       is_training=is_training)

        # average pooling
        x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)

        return y

