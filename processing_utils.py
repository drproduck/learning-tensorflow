# coding=utf-8
#prepossessing helpers for nn
import numpy as np

class batch_generator:
    def __init__(self,
                 data,
                 num_feature,
                 num_label,
                 label_index,
                 offset = 0):
        # assert data is np.ndarray
        self.data = data
        self.index = offset
        self.num_epoch_done = 0
        self.num_example = np.shape(data)[0]
        self.num_feature = num_feature
        self.num_label = num_label
        self.feature = data[:, :num_feature]
        self.label = np.zeros((self.num_example, num_label))
        for i in range(self.num_example):
            print(data[i][label_index])
            self.label[i][int(data[i][label_index])] = 1


    def next_batch(self, batch_size = 1):
        if self.num_example >= self.index + batch_size:
            if self.num_example == self.index + batch_size:
                self.num_epoch_done += 1
            self.index = (self.index + batch_size) % self.num_example
            return self.feature[self.index: self.index + batch_size], self.label[self.index: self.index + batch_size]
        else:
            num_first_half = self.num_example - self.index
            num_second_half = batch_size - num_first_half
            feature_first_half = self.feature[self.index:]
            feature_second_half = self.feature[:num_second_half]
            label_first_half = self.label[self.index:]
            label_second_half = self.label[:num_second_half]
            self.num_epoch_done += 1
            self.index = num_second_half
            return np.concatenate((feature_first_half, feature_second_half), axis=0), np.concatenate((label_first_half, label_second_half), axis=0)

    def reset(self):
        self.index = 0
        self.num_epoch_done = 0
















