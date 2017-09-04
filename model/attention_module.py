# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from keras.layers.convolutional import UpSampling2D

from .basic_layers import Conv, ResidualBlock


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

        self.first_residual_blocks = []
        for i in range(self.p):
            self.first_residual_blocks.append(ResidualBlock(self.input_channels))

        self.trunk_branches = []
        for i in range(self.t):
            self.trunk_branches.append(Conv([3, 3, input_channels, input_channels], strides=[1, 1, 1, 1]))

        self.second_residual_blocks = []
        for i in range(self.r):
            self.second_residual_blocks.append(ResidualBlock(self.input_channels))

        self.skip_connection_residual_block = ResidualBlock(self.input_channels)

        self.third_residual_blocks = []
        for i in range(self.r*2):
            self.third_residual_blocks.append(ResidualBlock(self.input_channels))

        self.forth_residual_blocks = []
        for i in range(self.r):
            self.forth_residual_blocks.append(ResidualBlock(self.input_channels))

        self.conv_1 = Conv([1, 1, self.input_channels, self.input_channels])
        self.conv_2 = Conv([1, 1, self.input_channels, self.input_channels])

        self.fifth_residual_blocks = []
        for i in range(self.p):
            self.fifth_residual_blocks.append(ResidualBlock(self.input_channels))

    def f_prop(self, x):
        """
        forward propagation
        :param x:
        :return:
        """

        ## first residual blocks
        for block in self.first_residual_blocks:
            x = block.f_prop(x)

        ## trunk branch
        for conv in self.trunk_branches:
            output_trunk = conv.f_prop(x)

        ## soft mask branch
        # max pooling
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')

        # residual blocks
        for block in self.second_residual_blocks:
            output_soft_mask = block.f_prop(output_soft_mask)

        # skip connection
        output_skip_connection = self.skip_connection_residual_block.f_prop(output_soft_mask)

        # max pooling
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

        # residual blocks
        for block in self.third_residual_blocks:
            output_soft_mask = block.f_prop(output_soft_mask)

        # interpolation TODO(check this method is valid)
        output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

        # add skip connection
        output_soft_mask += output_skip_connection

        # residual blocks
        for block in self.forth_residual_blocks:
            output_soft_mask = block.f_prop(output_soft_mask)

        # interpolation TODO(check this method is valid)
        output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

        # 1*1 conv
        output_soft_mask = self.conv1.f_prop(output_soft_mask)
        output_soft_mask = self.conv2.f_prop(output_soft_mask)

        # sigmoid
        output_soft_mask = tf.nn.sigmoid(output_soft_mask)

        ## attention
        output = (1 + output_soft_mask) * output_trunk

        ## last residual blocks
        for block in self.fifth_residual_blocks:
            output = block.f_prop(output)

        return output






