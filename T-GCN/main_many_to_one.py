# imports
import tensorflow as tf
import numpy as np
import numpy.linalg as la
from input_data import load_dow_price_data

# Data Dimensions
input_dim = 1           # input dimension
seq_max_len = 12         # sequence maximum length
out_dim = 1             # output dimension
seq_len = seq_max_len
pre_len = out_dim
train_rate = 0.8

# Parameters
learning_rate = 0.01    # The optimization initial learning rate
training_steps = 10000  # Total number of training steps
batch_size = 32         # batch size
display_freq = 1000     # Frequency of displaying the training results

num_hidden_units = 10   # number of hidden units


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


data, adj = load_dow_price_data()
time_len = data.shape[0]
num_nodes = data.shape[1]
data = data[:, 0]

# max_value = np.max(data)
# data = data/max_value
print("data.shape: ", data.shape)
x_train, y_train, x_valid, y_valid = preprocess_data(
    data, time_len, train_rate, seq_len, pre_len)

totalbatch = int(x_train.shape[0]/batch_size)
print("totalbatch: ", totalbatch)
# training_data_count = len(x_train)


def evaluation(a, b):
    F_norm = la.norm(a-b, 'fro')/la.norm(a, 'fro')
    return 1-F_norm


# ==========
#  TOY DATA
# ==========
x_train = np.random.randint(0, 10, size=(100, seq_max_len, 1))
y_train = np.sum(x_train, axis=1)

x_test = np.random.randint(0, 10, size=(5, seq_max_len, 1))
y_test = np.sum(x_test, axis=1)

print("Size of:")
print("- Training-set size:\t\t{}".format(len(y_train)))
print("- Test-set size:\t{}".format(len(y_test)))


# def next_batch(x, y, batch_size):
#     N = x.shape[0]
#     batch_indices = np.random.permutation(N)[:batch_size]
#     x_batch = x[batch_indices]
#     y_batch = y[batch_indices]
#     return x_batch, y_batch

# weight and bais wrappers


def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def RNN(x, weights, biases, num_hidden):
    """
    :param x: inputs of size [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param num_hidden: number of hidden units
    """
    cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    out = tf.matmul(outputs[:, -1, :], weights) + biases
    return out


# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)
x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])
y = tf.placeholder(tf.float32, [None, 1])
W = weight_variable(shape=[num_hidden_units, out_dim])
b = bias_variable(shape=[out_dim])

# Network predictions
pred_out = RNN(x, W, b, num_hidden_units)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
for i in range(totalbatch):
    m = i
    x_batch = x_train[m * batch_size: (m+1) * batch_size]
    y_batch = y_train[m * batch_size: (m+1) * batch_size]
    _, mse = sess.run([train_op, cost], feed_dict={x: x_batch, y: y_batch})
    if i % 10 == 0:
        print('Step {}, MSE={}'.format(i, mse))

# Test
y_pred = sess.run(pred_out, feed_dict={x: x_test})

for i, x in enumerate(y_test):
    print("When the ground truth output is {}, the model thinks it is {}"
          .format(y_test[i], y_pred[i]))
acc = evaluation(y_test, y_pred)
print("Accuracy: ", acc)
sess.close()
