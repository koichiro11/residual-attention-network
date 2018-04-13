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


class ResidualBlockDefault(object):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4

    """
    def __init__(self, kernel_size=3):
        """
        :param kernel_size: kernel size of second conv2d
        """
        self.kernel_size = kernel_size
        self._BATCH_NORM_DECAY = 0.997
        self._BATCH_NORM_EPSILON = 1e-5

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

    def batch_norm(self, inputs, is_training, data_format='channels_last',):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=True)

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def fixed_padding(self, inputs, kernel_size, data_format='channels_last'):
        """Pads the input along the spatial dimensions independently of input size.
        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                       Should be a positive integer.
          data_format: The input format ('channels_last' or 'channels_first').
        Returns:
          A tensor with the same format as the input with the data either intact
          (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs


class ResidualBlockBottleNeck(ResidualBlockDefault):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4

    """
    def __init__(self, kernel_size=3):
        """
        :param kernel_size: kernel size of second conv2d
        """
        super().__init__(kernel_size)

    def f_prop(self, inputs, filters, strides=1, scope="residual_block", is_training=True, data_format='channels_last'):
        """
        forward propagation
        pre-activation ResNets removing the fist ReLU with a BN layer after the final convolutional layer.
        https://arxiv.org/pdf/1610.02915.pdf
        :param inputs: A Tensor
        :param filters: dimension of output channel
        :param strides: int stride of kernel
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :param data_format: The input format ('channels_last' or 'channels_first').
        :return: output residual block
        """
        with tf.variable_scope(scope):
            shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=filters,
                                                 kernel_size=1, strides=1, data_format=data_format)
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters / 4, kernel_size=1, strides=1,
                data_format=data_format)
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs = tf.nn.relu(inputs)

            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters / 4, kernel_size=3, strides=strides,
                data_format=data_format)
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs = tf.nn.relu(inputs)

            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1,
                data_format=data_format)
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs += shortcut

            return inputs


class ResidualBlockWide(ResidualBlockDefault):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4

    """
    def __init__(self, kernel_size=3, widen=4):
        """
        :param kernel_size: kernel size of second conv2d
        """
        super().__init__(kernel_size)
        self.widen = widen

    def f_prop(self, inputs, filters, strides=1, scope="residual_block", is_training=True, data_format='channels_last'):
        """
        forward propagation
        widenet
        https://arxiv.org/pdf/1605.07146.pdf
        :param inputs: A Tensor
        :param filters: dimension of output channel
        :param strides: int stride of kernel
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :param data_format: The input format ('channels_last' or 'channels_first').
        :return: output residual block
        """
        with tf.variable_scope(scope):
            shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=filters * self.widen,
                                                 kernel_size=1, strides=1, data_format=data_format)

            # cnn3*3
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs = tf.nn.relu(inputs)
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters * self.widen, kernel_size=3, strides=1,
                data_format=data_format)

            # dropout
            inputs = tf.layers.dropout(inputs=inputs, rate=0.4, training=is_training)

            # cnn3*3
            inputs = self.batch_norm(inputs, is_training, data_format)
            inputs = tf.nn.relu(inputs)
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters * self.widen, kernel_size=3, strides=1,
                data_format=data_format)

            inputs += shortcut

            return inputs



