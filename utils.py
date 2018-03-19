# -*- coding: utf-8 -*-
"""
utils file
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from hyperparameter import HyperParams as hp

def load_data():
    if hp.target_dataset == "CIFAR-10":
        if os.path.exists(hp.DATASET_DIR + hp.target_dataset):
            print("load data from pickle")
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_X.pkl", 'rb') as f:
                train_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_y.pkl", 'rb') as f:
                train_y = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_X.pkl", 'rb') as f:
                valid_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_y.pkl", 'rb') as f:
                valid_y = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_X.pkl", 'rb') as f:
                test_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_y.pkl", 'rb') as f:
                test_y = pickle.load(f)
        else:
            (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
            cifar_X = np.r_[cifar_X_1, cifar_X_2]
            cifar_y = np.r_[cifar_y_1, cifar_y_2]

            cifar_X = cifar_X.astype('float32') / 255.0
            cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

            train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=5000,
                                                                random_state=hp.random_state)
            train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=5000,
                                                                  random_state=hp.random_state)

            os.mkdir(hp.DATASET_DIR + hp.target_dataset)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_X.pkl", 'wb') as f1:
                pickle.dump(train_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_y.pkl", 'wb') as f1:
                pickle.dump(train_y, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_X.pkl", 'wb') as f1:
                pickle.dump(valid_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_y.pkl", 'wb') as f1:
                pickle.dump(valid_y, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_X.pkl", 'wb') as f1:
                pickle.dump(test_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_y.pkl", 'wb') as f1:
                pickle.dump(test_y, f1)

    return train_X, train_y, valid_X, valid_y, test_X, test_y
