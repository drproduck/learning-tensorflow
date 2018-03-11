# coding=utf-8
# code to visualize the embeddings. uncomment the below to visualize embeddings
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# it has to variable. constants don't work here. you can't reuse model.embed_matrix
sess = tf.Session()
embedding_var = tf.Variable(mnist.test.images[:5000], name='embedding')
sess.run(embedding_var.initializer)
#
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter('visualization')
#
# add embedding to the config file
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
#
embedding.sprite.image_path = '/home/drproduck/Desktop/sprite.png'
embedding.sprite.single_image_dim.extend([28, 28])

# saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(summary_writer, config)
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, 'visualization/model.ckpt', 1)