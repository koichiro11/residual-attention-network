# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from tf.contrib.keras.layers import UpSampling2D

from basic_layers import Conv, ResidualBlock


class AttentionModule(object):
    """AttentionModuleClass"""
    def __init__(self, p=1, t=2, r=1):
        """
        :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
        :param t: the number of Residual Units in trunk branch
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch
        """
        self.p = p
        self.t = t
        self.r = r

    def f_prop(self, x):
        """
        forward propagation
        :param x:
        :return:
        """
        ## set channels
        input_channels = x.get_shape().as_list()[3]

        ## first residual blocks
        for i in range(self.p):
            residual_block = ResidualBlock()
            x = residual_block.f_prop(x)

        ## trunk branch
        for i in range(self.t):
            conv =Conv([3, 3, input_channels, input_channels], strides=[1, 1, 1, 1])
            output_trunk = conv.f_prop(x)

        ## soft mask branch
        # max pooling
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')

        # residual blocks
        for i in range(self.r):
            residual_block = ResidualBlock()
            output_soft_mask = residual_block.f_prop(output_soft_mask)

        # skip connection
        residual_block = ResidualBlock()
        output_skip_connection = residual_block.f_prop(output_soft_mask)

        # max pooling
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

        # residual blocks
        for i in range(self.r*2):
            residual_block = ResidualBlock()
            output_soft_mask = residual_block.f_prop(output_soft_mask)

        # interpolation TODO(check this method is valid)
        output_soft_mask = UpSampling2D([2,2])(output_soft_mask)

        # add skip connection
        output_soft_mask += output_skip_connection

        # residual blocks
        for i in range(self.r):
            residual_block = ResidualBlock()
            output_soft_mask = residual_block.f_prop(output_soft_mask)

        # interpolation TODO(check this method is valid)
        output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

        _channels = output_soft_mask.get_shape().as_list()[3]

        # 1*1 conv
        conv1 = Conv([1, 1, _channels, input_channels])
        output_soft_mask = conv1.f_prop(output_soft_mask)
        conv2 = Conv([1, 1, input_channels, input_channels])
        output_soft_mask = conv2.f_prop(output_soft_mask)

        # sigmoid
        output_soft_mask = tf.nn.sigmoid(output_soft_mask)

        ## attention
        output = (1 + output_soft_mask) * output_trunk

        ## last residual blocks
        for i in range(self.p):
            residual_block = ResidualBlock()
            output = residual_block.f_prop(output)

        return output






