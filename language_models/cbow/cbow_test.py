# coding=utf-8



VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 300  # dimension of the word embedding vectors
NUM_SAMPLED = 128  # Number of negative examples to sample.
LEARNING_RATE = 0.1
NUM_TRAIN_STEPS = 100000
SKIP_STEP = 2000

from language_models.cbow.batch_generator import *
words = read_data('text8.zip')

if __name__ == '__main__':
    model = cbow(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE, SKIP_STEP, window=4)
    model.build()
    data = read_data('text8.zip')
    word_to_index = word_to_index(data, VOCAB_SIZE)
    bat_gen = batch_generator(word_to_index, offset=0)
    model.train(bat_gen, NUM_TRAIN_STEPS)
