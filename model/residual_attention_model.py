# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import numpy as np
import tensorflow as tf

from basic_layers import Dense, Conv, ResidualBlock
from attention_module import AttentionModule


class ResidualAttentionModel(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self, input_channels, output_dim):
        self.output_dim = output_dim

    def f_prop(self, x):
        """
        forward propagation
        :param x: input x
        :return: outputs of probabilities
        """

        # set channels
        input_channels = x.get_shape().as_list()[3]
        conv1 = Conv([7, 7, input_channels, 64], strides=[1, 2, 2, 1])
        x = conv1.f_prop(x)

        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # residual block
        residual_block = ResidualBlock()
        x = residual_block.f_prop(x)

        # attention module
        attention_module = AttentionModule()
        x = attention_module.f_prop(x)

        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # residual block
        input_channels = x.get_shape().as_list()[3]
        residual_block = ResidualBlock(output_channels=input_channels*2)
        x = residual_block.f_prop(x)

        # attention module
        attention_module = AttentionModule()
        x = attention_module.f_prop(x)

        # residual block
        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        input_channels = x.get_shape().as_list()[3]
        residual_block = ResidualBlock(output_channels=input_channels*2)
        x = residual_block.f_prop(x)

        # attention module
        attention_module = AttentionModule()
        x = attention_module.f_prop(x)

        # max pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # residual block
        input_channels = x.get_shape().as_list()[3]
        residual_block = ResidualBlock(output_channels=input_channels*2)
        x = residual_block.f_prop(x)

        # average pooling
        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')

        # FC, softmax
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        dense = Dense([x.get_shape().as_list()[1], self.output_dim])
        y = dense.f_prop(x)

        return y

