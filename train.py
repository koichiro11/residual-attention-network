# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import numpy as np
import joblib
from model.utils import EarlyStopping
from model.residual_attention_network import ResidualAttentionNetwork

from preprocessor import PreProcessorWithAugmentation as Preprocess
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from tqdm import tqdm

from hyperparameter import HyperParams as hp


if __name__ == "__main__":
    print("start to train ResidualAttentionModel.")
    info = joblib.load(hp.SAVE_DIR / 'info.pkl')

    print("define preprocessor...")
    preprocess = Preprocess()

    # load dataset
    print("load TFRecord...")
    train_path = str(hp.SAVE_DIR / 'train*.tfrecord')
    valid_path = str(hp.SAVE_DIR / 'valid*.tfrecord')
    test_path = str(hp.SAVE_DIR / 'test*.tfrecord')
    image_size = info['image_size']
    train_dataset = preprocess.load_tfrecords_dataset(train_path, image_size, 10)
    valid_dataset = preprocess.load_tfrecords_dataset(train_path, image_size, 10)
    test_dataset = preprocess.load_tfrecords_dataset(test_path, image_size, 10)

    # get iterator
    print("get iterator...")
    aug_kwargs_train = {
        'resize_h': 40,
        'resize_w': 40,
        'input_h': 32,
        'input_w': 32,
        'channel': 3,
        'is_training': True,
    }
    train_iterator = preprocess.get_iterator(
        train_dataset, batch_size=hp.BATCH_SIZE, num_epochs=hp.NUM_EPOCHS, buffer_size=100 * hp.BATCH_SIZE,
        aug_kwargs=aug_kwargs_train)
    aug_kwargs_valid = {
        'resize_h': 40,
        'resize_w': 40,
        'input_h': 32,
        'input_w': 32,
        'channel': 3,
        'is_training': False,
    }
    valid_iterator = preprocess.get_iterator(
        train_dataset, batch_size=hp.VALID_BATCH_SIZE, num_epochs=hp.NUM_EPOCHS, buffer_size=100 * hp.VALID_BATCH_SIZE,
        aug_kwargs=aug_kwargs_valid)


    test_iterator = preprocess.get_iterator(
        test_dataset, batch_size=hp.VALID_BATCH_SIZE, num_epochs=hp.NUM_EPOCHS, buffer_size=100 * hp.VALID_BATCH_SIZE,
        aug_kwargs=aug_kwargs_valid)
    train_batch = train_iterator.get_next()
    valid_batch = valid_iterator.get_next()
    test_batch = test_iterator.get_next()

    print("build graph...")
    model = ResidualAttentionNetwork()
    early_stopping = EarlyStopping(limit=30)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    t = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, shape=())

    y = model.f_prop(x)

    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
    loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-7), reduction_indices=[1]))
    train = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean(loss))
    valid = tf.argmax(y, 1)

    print("start to train...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(hp.NUM_EPOCHS):
            n_batches = info["data_size"]["train"] // hp.BATCH_SIZE
            # train
            train_costs = []
            for i in tqdm(range(n_batches)):
                train_X_mb, train_y_mb = sess.run(train_batch)
                _, _loss = sess.run([train, loss], feed_dict={x: train_X_mb, t: train_y_mb, is_training: True})
                train_costs.append(_loss)

            # valid
            valid_costs = []
            valid_predictions = []
            valid_label = []
            n_batches = info["data_size"]["valid"] // hp.VALID_BATCH_SIZE
            for i in range(n_batches):
                valid_X_mb, valid_y_mb = sess.run(valid_batch)
                pred, valid_cost = sess.run([valid, loss], feed_dict={x: valid_X_mb, t: valid_y_mb, is_training: False})
                valid_predictions.extend(pred)
                valid_label.extend(np.argmax(valid_y_mb, 1).astype('int32'))
                valid_costs.append(valid_cost)

            # f1_score = f1_score(np.argmax(valid_y, 1).astype('int32'), valid_predictions, average='macro')
            accuracy = accuracy_score(valid_label, valid_predictions)
            if epoch % 5 == 0:
                print('EPOCH: {epoch}, Training cost: {train_cost}, Validation cost: {valid_cost}, Validation Accuracy: {accuracy} '
                      .format(epoch=epoch, train_cost=np.mean(train_costs), valid_cost=np.mean(valid_costs), accuracy=accuracy))

            if early_stopping.check(np.mean(valid_costs)):
                break

        print("save model...")
        saver = tf.train.Saver()
        saver.save(sess, hp.DATASET_DIR / 'model.ckpt', global_step=epoch)
