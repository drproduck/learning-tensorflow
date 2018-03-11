# coding=utf-8

import tensorflow as tf



def layer(x, weight, bias):
    w_init = tf.random_normal_initializer()
    b_init = tf.constant_initializer()
    stat_init= tf.random_normal_initializer()

    w = tf.get_variable('w',
                        shape=weight,
                        initializer=w_init)
    b = tf.get_variable('b',
                        shape=bias,
                        initializer=b_init)
    mean = tf.get_variable('mean',
                           shape=x.get_shape,
                           initializer=stat_init)
    variance = tf.get_variable('variance',
                           shape=x.get_shape,
                           initializer=stat_init)

    return tf.nn.relu(tf.nn.batch_normalization(tf.matmul(x, w) + b, mean=mean, variance=variance, offset=True, scale=True, variance_epsilon=))

def inference()
