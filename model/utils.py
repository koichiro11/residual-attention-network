# -*- coding: utf-8 -*-
"""
utils
"""


class EarlyStopping(object):
    """early stopping"""

    def __init__(self, limit=15):
        self.stop_count = 0
        self.limit = limit
        self.best_validation_loss = float('inf')

    def check(self, loss):
        if loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.stop_count = 0
        else:
            self.stop_count += 1

        if self.stop_count > self.limit:
            return True
        else:
            return False


def loss_filter(name):
    """
    select tensorflow variable for l2 regularization
    :param name:
    :return:
    """
    ng_scopes = ['batch_norm', 'bias']
    for ng_scope in ng_scopes:
        if ng_scope in name:
            return False
    return True