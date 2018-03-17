# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from keras.layers.convolutional import UpSampling2D

from .basic_layers import ResidualBlock


class AttentionModule(object):
    """AttentionModuleClass"""
    def __init__(self, input_channels, p=1, t=2, r=1):
        """
        :param input_channels: dimension of input channel.
        :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
        :param t: the number of Residual Units in trunk branch
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch
        """
        self.input_channels = input_channels
        self.p = p
        self.t = t
        self.r = r

        self.residual_block = ResidualBlock(input_channels)


    def f_prop(self, input, scope="attention_module", is_training=True):
        """
        f_prop function of attention module
        :param input: A Tensor. input data [batch_size, height, width, channel]
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: A Tensor [batch_size, height, width, channel]
        """
        with tf.variable_scope(scope):

            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(self.p):
                    input = self.residual_block.f_prop(input, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(self.t):
                    output_trunk = self.residual_block.f_prop(output_trunk, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.varable_scope("soft_mask_branch"):

                with tf.varable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

                    for i in self.r:
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.varable_scope("skip_connection"):
                    # TODO(define new blocks)
                    output_skip_connection = self.residual_block.f_prop(output_soft_mask, is_training=is_training)


                with tf.varable_scope("down_sampling_2"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

                    for i in self.r:
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.varable_scope("up_sampling_1"):
                    for i in self.r:
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.varable_scope("up_sampling_2"):
                    for i in self.r:
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


                with tf.variable_scope("output"):
                    output_soft_mask = tf.layers.conv1d(output_soft_mask, filters=self.input_channels, kernel_size=1)
                    output_soft_mask = tf.layers.conv1d(output_soft_mask, filters=self.input_channels, kernel_size=1)

                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(self.p):
                    output = self.residual_block.f_prop(output, scope="num_blocks_{}".format(i), is_training=is_training)

            return output
