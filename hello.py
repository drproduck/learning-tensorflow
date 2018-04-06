# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
x_data = np.random.uniform(1,10,1000)
y_data = x_data ** 3

x = tf.placeholder(tf.float32, [None, 784], 'input')

y = tf.placeholder(tf.float32, [None, 10], 'label')

w1 = tf.Variable(tf.random_normal([784, 256], stddev=(2.0/(784*256))**0.5))

b1 = tf.Variable(tf.constant(0, shape=[256]))

w2 = tf.Variable(tf.random_normal([256, 10], stddev=(2.0/(256*10))**0.5))

b2 = tf.Variable(tf.constant(0, [10]))

keep_prob = tf.placeholder(tf.float32)

output = tf.nn.relu(tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), keep_prob=keep_prob), w2) + b2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

sess = tf.Session()

global_step = tf.Variable(0, name='global_step', trainable=False)

sess.run(tf.initialize_all_variables)

num_epoch = 100
learn_rate = 0.01
batch_size = 128
num_batch = int(mnist.train.num_examples/128)

train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss, global_step=global_step)

for epoch in range(num_epoch):
    average_cost = 0
    for i in range(num_batch):
        images, labels = mnist.train.next_batch(batch_size)
        minibatch_cost, _ = sess.run((loss, train_op), feed_dict={x: images, y: images, keep_prob: 0.5})
        average_cost += minibatch_cost/num_batch

    print('epoch ', epoch)
    print('average cost', average_cost)

    validation_loss = sess.run(loss, feed_dict={x: mnist.validation.images, y: mnist.validation.images})

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('validation error', 1 - accuracy)
