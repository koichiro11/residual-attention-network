# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import UpSampling2D
import numpy as np


# 入力
# 入力 (4次元)
x = tf.placeholder(tf.float32)

# サンプル画像
sample_image = np.array([[1, 1, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0, 1],
                         [0, 0, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 0, 0, 1],
                         [0, 0, 1, 1, 1, 1],
                         ]
                       ).astype('float32').reshape(1, 6, 6, 1)

sample_image = np.array(np.random.randn(7,7)
                       ).astype('float32').reshape(1, 7, 7, 1)

# (バッチ, 行, 列, チャネル)

# フィルタ
W = np.array(np.random.randn(7,7)).astype('float32').reshape(7, 7, 1, 1)

W1 = np.array([[1, 0],
              [0, 1]]).astype('float32').reshape(2, 2, 1, 1)

convoluted_image = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

pooling_image = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

# shape = convoluted_image.get_shape()
"""
with tf.Session() as sess:
    # print(sess.run(convoluted_image, feed_dict={x: sample_image}).shape)
    print(sess.run(pooling_image, feed_dict={x: sample_image}).shape)
    # print(sess.run(shape, feed_dict={x: sample_image}))

    sess.close()
"""
print(UpSampling2D(size=(2, 2), data_format=None)(sample_image))
