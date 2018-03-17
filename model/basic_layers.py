# -*- coding: utf-8 -*-
"""
base models of Residual Attention Network
"""

import tensorflow as tf


class Layer(object):
    """basic layer"""
    def __init__(self, shape):
        """
        initial layer
        :param shape: shape of weight  (ex: [input_dim, output_dim]
        """
        # Xavier Initialization
        self.W = self.weight_variable(shape)
        self.b = tf.Variable(tf.zeros([shape[1]]))

    @staticmethod
    def weight_variable(shape, name=None):
        """define tensorflow variable"""
        # 標準偏差の2倍までのランダムな値で初期化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def f_prop(self, x):
        """forward propagation"""
        return tf.matmul(x, self.W) + self.b


class Dense(Layer):
    """softmax layer """
    def __init__(self, shape, function=tf.nn.softmax):
        """
        :param shape: shape of weight (ex:[input_dim, output_dim]
        :param function: activation ex:)tf.nn.softmax
        """
        super().__init__(shape)
        # Xavier Initialization
        self.function = function

    def f_prop(self, x):
        """forward propagation"""
        return self.function(tf.matmul(x, self.W) + self.b)


class ResidualBlock(Layer):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4

    """
    def __init__(self, input_channels, output_channels=None, stride=1, kernel_size=3):
        """
        :param input_channels: dimension of input channel.
        :param output_channels: dimension of output channel. input_channel -> output_channel
        """
        self.input_channels = input_channels
        if output_channels is not None:
            self.output_channels = output_channels
        else:
            self.output_channels = input_channels
        self.stride = stride
        self.kernel_size = kernel_size

    def f_prop(self, _input, scope="residual_block", is_training=True):
        """
        forward propagation
        :param _input: A Tensor
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: output residual block
        """
        with tf.variable_scope(scope):

            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_normalization(_input, self.input_channels, is_training)

            x = tf.layers.conv2d(x, filters=self.input_channels, kernel_size=self.kernel_size)

            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_normalization(x, self.input_channels, is_training)

            x = tf.layers.conv2d(x, filters=self.input_channels, kernel_size=self.kernel_size, strides=self.stride)

            # update input
            if (self.input_channels != self.output_channels) or (self.stride!=1):
                _input = tf.layers.conv1d(_input, self.output_channels, strides=self.stride)

            output = x + _input

            return output

    def batch_normalization(self,
                            x,
                            channels,
                            variance_epsilon=0.001,
                            scale_after_normalization=True,
                            scope="batch_norm",
                            is_training=True):
        """
        batch normalization
        :param x: input x
        :param channels: channels of input data
        :param variance_epsilon:  A small float number to avoid dividing by 0
        :param scale_after_normalization: A bool indicating whether the resulted tensor needs to be multiplied with gamma
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: batch normalized x
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.zeros([channels]), name="beta")
            gamma = self.weight_variable([channels], name="gamma")
            if is_training:
                mean, var = tf.nn.moments(x, axes=[0, 1, 2])

                x = tf.nn.batch_norm_with_global_normalization(
                    x, mean, var, beta, gamma, variance_epsilon,
                    scale_after_normalization=scale_after_normalization)
            # relu
            return tf.nn.relu(x)

