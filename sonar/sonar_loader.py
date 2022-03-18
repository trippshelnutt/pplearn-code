"""
Loader for sonar data testing mines vs rocks
"""

import numpy as np
import torch as t


def load_data(filename):
    return np.loadtxt(filename, delimiter=",", dtype=str)


def encode_mines(Y):
    return (Y == "M").astype(int)


data_without_bias = load_data("data/sonar.all-data")
data = np.insert(data_without_bias, 0, 1, axis=1)
np.random.seed(12345)
np.random.shuffle(data)

X = t.as_tensor(data[:, 0:-1].astype(np.float32))
Y = t.as_tensor(encode_mines(data[:, -1:]))

SIZE_OF_TRAINING_SET = 160

X_train, X_test = t.vsplit(X, [SIZE_OF_TRAINING_SET])
# print(X_train)
# print(X_train.shape)
# print(X_test)
# print(X_test.shape)

Y_train, Y_test = t.vsplit(Y, [SIZE_OF_TRAINING_SET])
# print(Y_train)
# print(Y_train.shape)
# print(Y_test)
# print(Y_test.shape)
