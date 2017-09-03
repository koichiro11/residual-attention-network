# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import sys
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.datasets import cifar10

from model.residual_attention_model import ResidualAttentionModel


rng = np.random.RandomState(1234)
random_state = 42

if __name__ == "__main__":
    print("start to train ResidualAttentionModel")

    param = sys.argv
    if len(param) == 2:
        target_dataset = param[1]
        if target_dataset == "CIFER-10":
            raise ValueError("Now you can use only 'CIFER-10' for training. "
                             "Please specify valid DataSet {'CIFER-10'} or "
                             "write build_model method in ResidualAttentionModel class by yourself.")

    else:
        target_dataset = "CIFER-10"

    print("load {dataset} data...".format(dataset=target_dataset))
    if target_dataset == "CIFER-10":
        (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
        cifar_X = np.r_[cifar_X_1, cifar_X_2]
        cifar_y = np.r_[cifar_y_1, cifar_y_2]

        cifar_X = cifar_X.astype('float32') / 255
        cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

        train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=10000,
                                                            random_state=random_state)
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=10000,
                                                              random_state=random_state)
    elif target_dataset == "ImageNet":
        # TODO(load DataSet for ImageNet)
        print("TODO(load DataSet for ImageNet)")

    else:
        raise ValueError("Now you can use only 'CIFER-10' for training. "
                         "Please specify valid DataSet {'CIFER-10'} or "
                         "write build_model method in ResidualAttentionModel class by yourself.")

    print("build graph...")
    model = ResidualAttentionModel()
    model.build_model(target=target_dataset)




