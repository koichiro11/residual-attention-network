# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf

from basic_layers import Dense
from attention_module import AttentionModule


class ResidualAttentionModel(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self, input_shape, output_dim):
        self.output_dim = 10

    def f_prop(self, x):
        """
        forward propagation
        :param x: input x
        :return: outputs of probabilities
        """

        # TODO(check image size)
