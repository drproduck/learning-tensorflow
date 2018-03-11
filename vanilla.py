import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
momentum = 0.9
epochs = 100
batch_size = 128
display_step = 1

def inference(x):
    with tf.variable_scope('hidden_1'):
        hidden_1 = layer(x, [784, 256], [256])

    with tf.variable_scope('hidden_2'):
        hidden_2 = layer(hidden_1, [256, 256], [256])

    with tf.variable_scope('output'):
        output = layer(hidden_2, [256, 10], [10])

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('validation error', accuracy)
    return accuracy

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
    initializer=w_init)
    b = tf.get_variable("b", bias_shape,
    initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

with tf.Graph().as_default():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder('float', [None, 784])

    y = tf.placeholder('float', [None, 10])

    output = inference(x)

    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)

    eval_op = evaluate(output, y)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('logistic_logs/', graph_def=sess
                                           .graph_def)

    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)

            feed_dict = {x: mbatch_x, y: mbatch_y}

            sess.run(train_op, feed_dict=feed_dict)

            minibatch_cost = sess.run(cost, feed_dict=feed_dict)

            avg_cost += minibatch_cost/total_batch

        if epoch % display_step == 0:
            val_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}

            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)

            print("validation error:", (1 - accuracy))

            summary_str = sess.run(summary_op, feed_dict)

            summary_writer.add_summary(summary_str, sess.run(global_step))

            saver.save(sess, 'logistic_logs/model-checkpoint', global_step=global_step)

    print('finished')

    test_feed_dict = {x: mnist.test.images, y: mnist.test.labels}

    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)

    print('test accuracy:', accuracy)




