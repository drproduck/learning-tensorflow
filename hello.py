# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def layer(x, weight_shape, bias_shape, scope_name):
    w_stddev = (2.0/weight_shape.shape[0]) ** 0.5
    w_init = tf.random_normal_initializer(stddev=w_stddev)
    b_init = tf.constant_initializer(value=0)
    stat_init= tf.random_normal_initializer()

    w = tf.get_variable('w',
                        shape=weight_shape,
                        initializer=w_init)
    b = tf.get_variable('b',
                        shape=bias_shape,
                        initializer=b_init)
    mean = tf.get_variable('mean',
                           shape=x.get_shape,
                           initializer=stat_init)
    variance = tf.get_variable('variance',
                               shape=x.get_shape,
                               initializer=stat_init)

    return tf.nn.relu(tf.nn.batch_normalization(tf.matmul(x, w) + b, mean=mean, variance=variance, offset=True, scale=True, variance_epsilon=1e-12))

def loss(output, y):
    # xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    # loss = tf.reduce_mean(xentropy)
    loss = tf.reduce_mean(tf.square(tf.abs(output - y)))
    return loss

def training(cost, global_step, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output_test, y_test):
    correct_prediction = tf.equal(tf.argmax(output_test, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def main():
    x_data = np.random.uniform(1,10,1000).reshape(1000)[:,None]
    print(x_data[0:10,0])
    y_data = x_data ** 3
    x_test = np.random.uniform(10,15,20).reshape(20)[:,None]
    y_test = x_test ** 3
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float64, shape=[None, 1])
        y = tf.placeholder(tf.float64, shape=[None, 1])
        w1 = tf.Variable(tf.random_normal(shape=[1,128], stddev=0.5, dtype=tf.float64), dtype=tf.float64)
        b1 = tf.Variable(tf.random_normal(shape=[128], dtype=tf.float64), dtype=tf.float64)
        w2 = tf.Variable(tf.random_normal(shape=[128,1], dtype=tf.float64), dtype=tf.float64)
        b2 = tf.Variable(tf.random_normal(shape=[1], dtype=tf.float64), dtype=tf.float64)
        output = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2)
        cost = loss(output, y)
        global_step = tf.Variable(0, name='step', trainable=False)
        learning_rate = tf.placeholder(dtype=tf.float32)
        train_op = training(cost, global_step=global_step, learning_rate=learning_rate)
        eval_op = evaluate(output, y)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        num_epoch = 5000
        learn_rate = 0.1

        for epoch in range(num_epoch):
            average_cost = 0
            cost_value, _ = sess.run((cost, train_op), feed_dict={x: x_data, y: y_data, learning_rate: learn_rate})

            print('epoch ', epoch)
            print('cost', cost_value)

        output_value = sess.run(output, feed_dict={x: x_test})
        # print(np.concatenate((x_test[1:100], output_value[1:100], y_test[1:100]), axis=1))
        # print('test accuracy = ',accuracy)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(211)
        plt.plot(np.arange(0,20), y_test)
        plt.subplot(212)
        plt.plot(np.arange(0,20), output_value)
        plt.show()

if __name__ == '__main__':
    main()