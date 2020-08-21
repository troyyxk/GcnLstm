# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import load_dow_price_data
from tgcn import tgcnCell
#from gru import GRUCell

from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers

data, adj, label = load_dow_price_data()

lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var)
                         for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
loss = tf.reduce_mean(tf.nn.l2_loss(pred-label) + Lreg)
error = tf.sqrt(tf.reduce_mean(tf.square(pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    pre_len = 1
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


time_len = data.shape[0]
num_nodes = data.shape[1]

trainX, trainY, testX, testY = preprocess_data(
    data, time_len, 0.8, 12, 1)

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model = build_model(allow_cudnn_kernel=False)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)
model.fit(
    x_train, y_train, validation_data=(
        x_test, y_test), batch_size=batch_size, epochs=1
)
