# -*- coding: utf-8 -*-
"""
Wide Residual Networks
https://arxiv.org/pdf/1605.07146.pdf
"""

import tensorflow as tf
import numpy as np

from .basic_layers import ResidualBlock


class WideResidualNetworks(object):
    """
    Residual Attention Network for cifar-10.
    If you would like to use anther dataset, please override this class
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        self.input_shape = [-1, 32, 32, 3]
        self.output_dim = 10

        # for cifar-10, you should use attention module 2 for first stage
        self.N = 6
        self.k = 10
        self.residual_block = ResidualBlock()

    def f_prop(self, x, is_training=True):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        # conv1, x -> [None, row, line, 16]
        with tf.variable_scope("conv_1"):
            x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='SAME')

            # max pooling, x -> [None, row, line, 16]
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # conv2
        with tf.variable_scope("conv_2"):
            for i in range(self.N):
                x = self.residual_block.f_prop(x, 16 * self.k, is_resize=(True if i == 0 else False),
                                           scope="num_blocks_{}".format(i),
                                           is_training=is_training)


        # conv3
        with tf.variable_scope("conv_3"):
            for i in range(self.N):
                x = self.residual_block.f_prop(x, 32 * self.k, is_resize=(True if i == 0 else False),
                                               scope="num_blocks_{}".format(i),
                                               is_training=is_training)

            # max pooling, x -> [None, row/2, line/2, 32 * self.k]
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv4
        with tf.variable_scope("conv_4"):
            for i in range(self.N):
                x = self.residual_block.f_prop(x, 64 * self.k, is_resize=(True if i == 0 else False),
                                               scope="num_blocks_{}".format(i),
                                               is_training=is_training)

            # max pooling, x -> [None, row/2, line/2, 32 * self.k]
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # average pooling
        x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        # layer normalization
        # x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)

        return y

