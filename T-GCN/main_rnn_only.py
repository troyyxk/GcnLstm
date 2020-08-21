# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from input_data import load_dow_price_data
import numpy.linalg as la


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])  # seq_len 12
        trainY.append(a[seq_len: seq_len + pre_len])  # pre_len 1
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


# Data Dimension
num_input = 28          # MNIST data input (image shape: 28x28)
timesteps = 28          # Timesteps
n_classes = 1           # Number of classes, one class per digit

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 1, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 28, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
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

data, adj = load_dow_price_data()
time_len = data.shape[0]
num_nodes = data.shape[1]
data = data[:, 0]

max_value = np.max(data)
data = data/max_value
print("data.shape: ", data.shape)
x_train, y_train, x_valid, y_valid = preprocess_data(
    data, time_len, train_rate, seq_len, pre_len)

totalbatch = int(x_train.shape[0]/batch_size)
training_data_count = len(x_train)

learning_rate = 0.001  # The optimization initial learning rate
epochs = 10           # Total number of training epochs
batch_size = 100      # Training batch size
display_freq = 100    # Frequency of displaying the training results
num_hidden_units = 128  # Number of hidden units of the RNN


def weight_variable(shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def RNN(x, weights, biases, timesteps, num_hidden):
    x = tf.unstack(x, timesteps, 1)
    rnn_cell = rnn.BasicRNNCell(num_hidden)
    _, current_state = rnn.static_rnn(
        rnn_cell, x, dtype=tf.float32)
    return tf.matmul(current_state, weights) + biases


# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, timesteps, 1], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
W = weight_variable(shape=[num_hidden_units, n_classes])
b = bias_variable(shape=[n_classes])

output_logits = RNN(x, W, b, timesteps, num_hidden_units)
y_pred = output_logits

lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var)
                         for tf_var in tf.trainable_variables())
label = tf.reshape(y, [-1, num_nodes])
# loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate, name='Adam-op').minimize(loss)


def evaluation(a, b):
    F_norm = la.norm(a-b, 'fro')/la.norm(a, 'fro')
    return 1-F_norm


# Creating the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    for iteration in range(num_tr_iter):
        m = iteration
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        print(x_train.shape)
        print(y_train.shape)

        x_batch = x_train[m * batch_size: (m+1) * batch_size]
        y_batch = y_train[m * batch_size: (m+1) * batch_size]

        x_batch = x_batch.reshape(-1, timesteps)
        x_batch = x_batch.reshape(-1, timesteps, 1)

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            test_output = sess.run([output_logits],
                                   feed_dict=feed_dict_batch)

            acc = evaluation(
                y_batch, test_output)
            print("iter {0:3d}:\t Training Accuracy={2:.01%}".
                  format(iteration, acc))

    # Run validation after every epoch

    # feed_dict_valid = {x: x_valid[:1000].reshape(
    #     (-1, timesteps, num_input)), y: y_valid[:1000]}
    # loss_valid, acc_valid = sess.run(
    #     [loss, accuracy], feed_dict=feed_dict_valid)
    # print('---------------------------------------------------------')
    # print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
    #       format(epoch + 1, loss_valid, acc_valid))
    # print('---------------------------------------------------------')

# Test the network (only on 1000 samples) after training
# Accuracy
# x_test, y_test = load_data(mode='test')
# feed_dict_test = {x: x_test[:1000].reshape(
#     (-1, timesteps, num_input)), y: y_test[:1000]}
# loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
# print('---------------------------------------------------------')
# print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(
#     loss_test, acc_test))
# print('---------------------------------------------------------')
