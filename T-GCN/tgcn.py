# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from utils import calculate_laplacian


class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, num_features, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes  # GCN
        self._units = num_units  # GRU, 64
        self._features = num_features  # 71
        self._adj = []
        self._adj.append(calculate_laplacian(adj))

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):
        print("-----------------inputs, state------------------------")
        print(inputs)
        print(state)
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                print("(((((((((((((((((gates)))))))))))")
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
                # u, r = tf.split(value=value, num_or_size_splits=2, axis=1)

                # r = tf.nn.sigmoid(
                #     self._gc(inputs, state, self._units, bias=1.0, scope=scope))
                # u = tf.nn.sigmoid(
                #     self._gc(inputs, state, self._units, bias=1.0, scope=scope))
            with tf.variable_scope("candidate"):
                print("(((((((((((((((((candidate)))))))))))")
                r_state = r * state
                c = self._act(self._gc(inputs, r_state,
                                       self._units, scope=scope))
                print(r_state)
                print("------------r_state-----------------")
                print("r: ", r.get_shape())
                print("u: ", u.get_shape())
                print("c: ", c.get_shape())
                print("state: ", state.get_shape())
                print("r_state: ", r_state.get_shape())
                print("-------------------------------")
                print("\n")
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        # inputs:(-1,num_nodes)
        inputs = tf.transpose(inputs, perm=[0, 2, 1])
        print("------------test expand_dims---------------")
        print("input before: ", inputs)
        inputs = tf.expand_dims(inputs, 3)
        print("input after: ", inputs)
        # state:(batch,num_node,gru_units)
        print("-----------test state reshape-----------")
        print("state before: ", state)
        state = tf.reshape(
            state, (-1, self._features, self._nodes, self._units))
        # state = tf.reshape(
        #     state, (-1, self._nodes, self._units, self._features))
        print("state after: ", state)
        # concat
        # TODO, original
        # x_s = tf.concat([inputs, state], axis=2)
        x_s = tf.concat([state, inputs], axis=3)

        input_size = x_s.get_shape()[3].value
        x0 = tf.transpose(x_s, perm=[2, 1, 3, 0])
        # gcn, features, gru, ?
        tmp = x0
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        print("------------x_s input_size-----------------")
        print("x_s: ", x_s.get_shape())  # ?*207*65
        print("tmp: ", tmp.get_shape())  # 207*65*?
        print("x0: ", x0.get_shape())  # 207*?
        print("input_size: ", input_size)  # 65
        print("-------------------------------")
        print("\n")

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            #             for m in self._adj:
            #                 # a = m
            #                 x1 = tf.sparse_tensor_dense_matmul(m, x0)
            # #                print(x1)

            x1 = tf.sparse_tensor_dense_matmul(self._adj[0], x0)

            # no difference
            # because self._adj has only 1 element, which is the adjecency matrix, and self._adj[0] is 207*207
            # for m in self._adj:
            #     a = m
            # x1 = tf.sparse_tensor_dense_matmul(a, x0)

            x = tf.reshape(
                x1, shape=[self._nodes, self._features, input_size, -1])
            print("x: ", x.get_shape())
            # x = x[:, 0:input_size, :]

            print("------------shape------------")
            print("x: ", x.get_shape())
            print("x0: ", x0.get_shape())
            print("x1: ", x1.get_shape())
            print(np.array(self._adj).shape)
            # print(a.shape)
            print("-----------------------------")

            x = tf.transpose(x, perm=[3, 1, 0, 2])
            # ?, features, gcn, gru
            print("-------------before weight----------------------")
            print("x: ", x)
            print("x.shape: ", x.shape)
            print("x.get_shape(): ", x.get_shape())
            tmp2 = x
            x = tf.reshape(x, shape=[-1, input_size])
            # print()

            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            # (batch_size * self._nodes, output_size)
            x = tf.matmul(x, weights)
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.nn.bias_add(x, biases)
            print("------------- Output size----------------------")
            print("x: ", x)
            print("x.shape: ", x.shape)
            print("x.get_shape(): ", x.get_shape())
            # ?, features, gcn, gru
            x = tf.reshape(
                x, shape=[-1, self._features, self._nodes, output_size])
            x = tf.reshape(
                x, shape=[-1, self._nodes * output_size])
        return x
