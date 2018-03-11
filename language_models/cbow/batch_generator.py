# coding=utf-8
import zipfile
from operator import itemgetter
import numpy as np

import tensorflow as tf
from collections import Counter

def read_data(file_path):
    """ Read data into a list of tokens
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # tf.compat.as_str() converts the input into the string
    return words

def create_metadata(words, thres):
    counter = Counter(words)
    top_words = counter.most_common(thres-1)
    f = open('vocab.tsv', 'w')
    f.write('null\n')
    for word, freq in top_words:
        f.write(word+'\n')
    f.close()

def word_to_index(words, thres):
    counter = Counter(words)
    top_words = counter.most_common(thres-1)

    vocab = dict()
    vocab['null'] = 0
    index = 1

    for word, _ in top_words:
        vocab[word] = index
        index += 1

    to_index = [vocab[word] if word in vocab else 0 for word in words]
    return to_index

class batch_generator:
    def __init__(self, data, offset):
        self.data = data
        self.index = offset

    def set_index(self, offset):
        self.index = offset

    def next_batch(self, batch_size, window=4):
        input_batch = []
        target_batch = []
        for _ in range(batch_size):
            seq = self.get_context(window)
            input_words = np.concatenate((seq[:window], seq[window+1:]))
            target_word = seq[window:window+1]
            input_batch.append(input_words)
            target_batch.append(target_word)
        return input_batch, target_batch

    def get_context(self, window):
        if self.index+window*2+1 > len(self.data):
            start = self.index
            first = self.data[start:]
            end = window*2+1 - len(first)
            second = self.data[:end]
            if self.index + 1 > len(self.data): self.index = 0
            else: self.index += 1
            return np.concatenate((first, second))
        else:
            start = self.index
            end = start + window*2+1
            self.index += 1
            return self.data[start: end]


create_metadata(read_data('text8.zip'), thres=10000)
