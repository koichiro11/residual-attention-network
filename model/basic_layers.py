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


class ResidualBlock(object):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4

    """
    def __init__(self, kernel_size=3):
        """
        :param kernel_size: kernel size of second conv2d
        """
        self.kernel_size = kernel_size

    def f_prop(self, _input, input_channels, output_channels=None, scope="residual_block", is_training=True):
        """
        forward propagation
        :param _input: A Tensor
        :param input_channels: dimension of input channel.
        :param output_channels: dimension of output channel. input_channel -> output_channel
        :param stride: int stride of kernel
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: output residual block
        """
        if output_channels is None:
            output_channels = input_channels

        with tf.variable_scope(scope):
            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_norm(_input, input_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=1, padding='SAME', name="conv1")

            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_norm(x, output_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=self.kernel_size,
                                 strides=1, padding='SAME', name="conv2")

            # update input
            if input_channels != output_channels:
                _input = tf.layers.conv2d(_input, filters=output_channels, kernel_size=1, strides=1)

            output = x + _input

            return output

    @staticmethod
    def batch_norm(x, n_out, is_training=True):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('batch_norm'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(tf.cast(is_training, tf.bool),
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return tf.nn.relu(normed)


