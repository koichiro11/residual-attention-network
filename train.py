# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import numpy as np
import time

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from c import tqdm
import utils
from model.utils import EarlyStopping
from model.residual_attention_network import ResidualAttentionNetwork
from hyperparameter import HyperParams as hp


if __name__ == "__main__":
    print("start to train ResidualAttentionModel")
    train_X, train_y, valid_X, valid_y, test_X, test_y = utils.load_data()

    print("build graph...")
    model = ResidualAttentionNetwork()
    early_stopping = EarlyStopping(limit=30)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    t = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, shape=())

    y = model.f_prop(x)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
    # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean(loss))
    valid = tf.argmax(y, 1)

    print("check shape of data...")
    print("train_X: {shape}".format(shape=train_X.shape))
    print("train_y: {shape}".format(shape=train_y.shape))

    print("start to train...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(hp.NUM_EPOCHS):
            train_X, train_y = shuffle(train_X, train_y, random_state=hp.RANDOM_STATE)
            # batch_train_X, batch_valid_X, batch_train_y, batch_valid_y = train_test_split(train_X, train_y, train_size=0.8, random_state=random_state)
            n_batches = train_X.shape[0] // hp.BATCH_SIZE

            # train
            train_costs = []
            for i in tqdm(range(n_batches)):
                # print(i)
                start = i * hp.BATCH_SIZE
                end = start + hp.BATCH_SIZE
                _, _loss = sess.run([train, loss], feed_dict={x: train_X[start:end], t: train_y[start:end], is_training: True})
                train_costs.append(_loss)

            # valid
            valid_costs = []
            valid_predictions = []
            n_batches = valid_X.shape[0] // hp.VALID_BATCH_SIZE
            for i in range(n_batches):
                start = i * hp.VALID_BATCH_SIZE
                end = start + hp.VALID_BATCH_SIZE
                pred, valid_cost = sess.run([valid, loss], feed_dict={x: valid_X[start:end], t: valid_y[start:end], is_training: False})
                valid_predictions.extend(pred)
                valid_costs.append(valid_cost)

            # f1_score = f1_score(np.argmax(valid_y, 1).astype('int32'), valid_predictions, average='macro')
            accuracy = accuracy_score(np.argmax(valid_y, 1).astype('int32'), valid_predictions)
            if epoch % 5 == 0:
                print('EPOCH: {epoch}, Training cost: {train_cost}, Validation cost: {valid_cost}, Validation Accuracy: {accuracy} '
                      .format(epoch=epoch, train_cost=np.mean(train_costs), valid_cost=np.mean(valid_costs), accuracy=accuracy))

            if early_stopping.check(np.mean(valid_costs)):
                break

        print("save model...")
        saver = tf.train.Saver()
        saver.save(sess, hp.SAVE_PATH, global_step=epoch)
