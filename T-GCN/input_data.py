# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle as pkl
from os import listdir


def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'../data/sz_adj.csv', header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'../data/sz_speed.csv')
    return sz_tf, adj


def load_los_data(dataset):
    los_adj = pd.read_csv(r'../data/los_adj.csv', header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'../data/los_speed.csv')
    return los_tf, adj


def load_dow_price_data():
    dow_adj = pd.read_csv(r'../Data_stock/dow_corr.csv', header=None)
    adj = np.mat(dow_adj)
    dow_price = pd.read_csv(r'../Data_stock/dow_price.csv').iloc[1000: 7000]
    # time * gcn node

    return dow_price, adj


def load_dow_full_data():
    dow_adj = pd.read_csv(r'../Data_stock/dow_corr.csv', header=None)
    adj = np.mat(dow_adj)

    dow_files_addr = "../Data_stock/dow/"
    dow_files_names = [f for f in listdir(dow_files_addr)]

    dow_data = []
    for f in dow_files_names:
        print("Read file: ", f)
        # limit length
        dow_data.append(pd.read_csv(dow_files_addr + f).values[1000: 7000])
    dow_data = np.array(dow_data).transpose((1, 0, 2))
    # time * gcn node * features

    label = pd.read_csv(r'../Data_stock/dow_price.csv').values[1000: 7000]
    return dow_data, adj, label


def process_dow_full_data(data, label, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    train_label = label[0:train_size]
    test_label = label[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        b = train_label[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(b[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        a = test_data[i: i + seq_len + pre_len]
        b = test_label[i: i + seq_len + pre_len]
        testX.append(a[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


def preprocess_data(data, time_len, rate, seq_len, pre_len):
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
