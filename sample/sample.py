# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
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

# (バッチ, 行, 列, チャネル)

# フィルタ
W = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]]).astype('float32').reshape(3, 3, 1, 1)

W1 = np.array([[1, 0],
              [0, 1]]).astype('float32').reshape(2, 2, 1, 1)

convoluted_image = tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

with tf.Session() as sess:
    print(sess.run(convoluted_image, feed_dict={x: sample_image}))