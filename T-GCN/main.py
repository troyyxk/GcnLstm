# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import *
from tgcn import tgcnCell
#from gru import GRUCell

from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import matplotlib.pyplot as plt
import time

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 1, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'los', 'sz or los.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

##### load data ######
if data_name == 'sz':
    data, adj = load_sz_data('sz')
if data_name == 'los':
    data, adj = load_los_data('los')

# data, adj = load_dow_price_data()
# data, adj, label = load_dow_full_data()

time_len = data.shape[0]
num_nodes = data.shape[1]
# num_features = data.shape[2]
print("--------------data info--------------------")
print("time_len: ", time_len)
# print("num_nodes: ", num_nodes)

data1 = np.array(data, dtype=np.float32)


# print("--------------np.count_nonzero(data1)--------------------------")
# print(np.count_nonzero(data1))

# normalize
# data1 = np.mat(data, dtype=np.float32)/num_nodes


# normalization
max_value = np.max(data1)
# test result for before and after the normalization, do not normalize is better
# test with 1000:7000, price only
# min_rmse:4218.586651805764 min_mae:2829.233 max_acc:0.4141682982444763 r2:-0.06428861618041992 var:-0.0431898832321167
# min_rmse:4273.191624041597 min_mae:2862.866 max_acc:0.4065856337547302 r2:-0.09201860427856445 var:-0.0651127099990844
# min_rmse:4215.928421487219 min_mae:2828.4226 max_acc:0.4145374894142151 r2:-0.06294786930084229 var:-0.03952479362487793
data1 = data1/max_value
# min_rmse:26.51728604225631 min_mae:18.180794 max_acc:0.4408230781555176 r2:0.030353426933288574 var:0.03035348653793335

# normal process data
trainX, trainY, testX, testY = preprocess_data(
    data1, time_len, train_rate, seq_len, pre_len)

# trainX, trainY, testX, testY = process_dow_full_data(
#     data1, label, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)


def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    print("---------------len(outputs)----------------")
    print(len(outputs))
    print(type(outputs[0]))
    print(outputs[0])
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        # comment the line below makes no difference
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    # outputs = gru_units * number of nodes in gcn
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    # num_nodes * pre_len, 3 in this case
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    # pre_len * num_nodes
    return output, m, states


###### placeholders ######
# TODO, check if the input need to be reshape to add features
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}

if model_name == 'tgcn':
    pred, ttts, ttto = TGCN(inputs, weights, biases)

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var)
                         for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
# loss
loss = tf.reduce_mean(tf.nn.l2_loss(pred-label) + Lreg)
# rmse
error = tf.sqrt(tf.reduce_mean(tf.square(pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s' % (model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

###### evaluation ######


def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b, 'fro')/la.norm(a, 'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [
], [], [], [], [], [], []

for epoch in range(training_epoch):
    print("epoch: ", epoch)
    # totalbatch = 500
    for m in range(totalbatch):
        print("m/totalbatch: ", m, "/", totalbatch)
        mini_batch = trainX[m * batch_size: (m+1) * batch_size]
        mini_label = trainY[m * batch_size: (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label})
        # print("mini_batch.shape: ", mini_batch.shape)
        # print("mini_label.shape: ", mini_label.shape)
        # print("train_output.shape: ", train_output.shape)
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

    print("-1-")
    # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, pred],
                                         feed_dict={inputs: testX, labels: testY})
    print("-2-")
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print("-3-")

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))

    if (epoch % 500 == 0):
        saver.save(sess, path+'/model_100/TGCN_pre_%r' %
                   epoch, global_step=epoch)

time_end = time.time()
print(time_end-time_start, 's')

############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch)
              for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch)
              for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv', index=False, header=False)
# plot_result(test_result,test_label1,path)
# plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])
