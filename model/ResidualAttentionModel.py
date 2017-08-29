# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf


class ResidualAttentionModel(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self, num_attention_module=4, stride=2):
        self.num_attention_module = num_attention_module
        self.stride = stride

    @staticmethod
    def weight_variable(shape, name=None):
        """define tensorflow variable"""
        # 標準偏差の2倍までのランダムな値で初期化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def softmax_layer(self, input, shape):
        """
        softmax layer
        :param input: input x
        :param shape: shape of x
        :return: output of softmax
        """
        fc_w = self.weight_variable(shape)
        fc_b = tf.Variable(tf.zeros([shape[1]]))

        fc_h = tf.nn.softmax(tf.matmul(input, fc_w) + fc_b)

        return fc_h

    def conv_layer(self, input, filter_shape, stride):
        """
        convolution layer with batch normalization
        :param input: input x
        :param filter_shape: filter shape (ex:[row of filter, line of filter , input channel, output channel]
        :param stride: stride (ex: 2
        :return: output of convolution
        """
        out_channels = filter_shape[3]
        # convolution
        filter_ = self.weight_variable(filter_shape)
        conv = tf.nn.conv2d(input, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

        # batch normalization
        mean, var = tf.nn.moments(conv, axes=[0,1,2])
        beta = tf.Variable(tf.zeros([out_channels]), name="beta")
        gamma = self.weight_variable([out_channels], name="gamma")

        batch_norm = tf.nn.batch_norm_with_global_normalization(
            conv, mean, var, beta, gamma, 0.001,
            scale_after_normalization=True)

        out = tf.nn.relu(batch_norm)

        return out

    def residual_block(self, input, output_channels, projection=False):
        """
        residual block : proposed residual block by https://arxiv.org/pdf/1603.05027.pdf
        :param input: input x
        :param output_channels: output channels
        :param projection: how to deal with the difference between input channels and output channels
        :return: output of residual block
        """

        input_channels = input.get_shape().as_list()[3]

        # batch normalization
        mean, var = tf.nn.moments(input, axes=[0, 1, 2])
        beta = tf.Variable(tf.zeros([input_channels]), name="beta")
        gamma = self.weight_variable([input_channels], name="gamma")

        batch_norm = tf.nn.batch_norm_with_global_normalization(
            input, mean, var, beta, gamma, 0.001,
            scale_after_normalization=True)

        # relu
        batch_normed_input = tf.nn.relu(batch_norm)

        # convolution with batch normalization
        conv1 = self.conv_layer(batch_normed_input, [3, 3, input_channels, output_channels], 1)

        # convolution
        conv2 = tf.nn.conv2d(conv1, filter=[3, 3, output_channels, output_channels], strides=[1, 1, 1, 1], padding="SAME")

        if input_channels != output_channels:
            if projection:
                # Option B: Projection shortcut
                input_layer = self.conv_layer(input, [1, 1, input_channels, output_channels], self.stride)
            else:
                # Option A: Zero-padding
                input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_channels - output_channels]])
        else:
            input_layer = input

        res = conv2 + input_layer
        return res