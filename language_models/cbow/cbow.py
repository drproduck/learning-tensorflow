# coding=utf-8
import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


class cbow:

    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate=1, skip_step=2000, window=4, has_checkpoint=True, has_summary=True, has_visual=True):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.has_checkpoint = has_checkpoint
        self.has_summary = has_summary
        self.skip_step = skip_step
        self.has_visual = has_visual
        self.window = window

        with tf.device('/cpu:0'):
            self.index = tf.Variable(0, dtype=tf.int32, trainable=False, name='next_sample_index')
            self.previous_words = tf.placeholder(tf.int32, name='previous_words')
            self.next_word = tf.placeholder(tf.int32, name='next_word')
            self.embed_matrix = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.embed_size], name='embed_matrix'))

    def create_loss(self):
        #mean of embed
        with tf.device('/gpu:0'):
            embed = tf.reduce_mean(tf.nn.embedding_lookup(params=self.embed_matrix, ids=self.previous_words, name='embed'), axis=1)

        nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                     stddev=1.0 / (self.embed_size ** 0.5)),
                                 name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')

        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                  biases=nce_bias,
                                                  labels=self.next_word,
                                                  inputs=embed,
                                                  num_sampled=self.num_sampled,
                                                  num_classes=self.vocab_size), name='loss')

    def create_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.loss, global_step=self.global_step)

    def create_summary(self):
        if self.has_summary:
            with tf.name_scope("summaries"):
                tf.summary.scalar("loss", self.loss)
                tf.summary.histogram("histogram loss", self.loss)
                # because you have several summaries, we should merge them all
                # into one op to make it easier to manage
                self.summary_op = tf.summary.merge_all()

        else: self.summary_op = None

    def build(self):
        self.create_loss()
        self.create_optimizer()
        self.create_summary()

    def train(self, batch_generator, num_train_step):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path and self.has_checkpoint:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('board', sess.graph)

            initial_step = self.global_step.eval(sess)
            offset = self.index.eval(sess)
            batch_generator.set_index(offset)

            for index in range(initial_step, initial_step + num_train_step):
                previous_words, next_word = batch_generator.next_batch(batch_size=self.batch_size, window=self.window)
                feed_dict = {self.previous_words: previous_words, self.next_word: next_word}

                loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                  feed_dict=feed_dict)
                if summary is not None:
                    writer.add_summary(summary, global_step=index)
                total_loss += loss_batch
                offset += self.batch_size
                if (index + 1) % self.skip_step == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                    total_loss = 0.0
                    saver.save(sess, 'checkpoints/cbow', index)

                    if self.has_visual:
                        final_embed_matrix = sess.run(self.embed_matrix)

                        embedding_var = tf.Variable(final_embed_matrix[:5000], name='embedding')
                        sess.run(embedding_var.initializer)

                        config = projector.ProjectorConfig()
                        summary_writer = tf.summary.FileWriter('processed')

                        # add embedding to the config file
                        embedding = config.embeddings.add()
                        embedding.tensor_name = embedding_var.name

                        embedding.metadata_path = '/home/drproduck/Documents/deep_learning/language_models/vocab.tsv'

                        # saves a configuration file that TensorBoard will read during startup.
                        projector.visualize_embeddings(summary_writer, config)
                        saver_embed = tf.train.Saver([embedding_var])
                        saver_embed.save(sess, 'processed/visual.ckpt', 1)








