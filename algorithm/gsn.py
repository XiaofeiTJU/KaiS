import tensorflow as tf
import numpy as np


def glorot(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        init = tf.random_uniform(
            shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init)


def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.zeros(shape, dtype=dtype)
        return tf.Variable(init)


class GraphSNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim, act_fn, scope='gsn'):
        self.inputs = inputs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims
        self.act_fn = act_fn
        self.scope = scope
        self.summ_levels = 2

        # initialize summarization parameters
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.global_weights, self.global_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        # graph summarization operation
        self.summaries = self.summarize()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        weights = []
        bias = []
        curr_in_dim = input_dim
        # Hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim
        # Output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))
        return weights, bias

    def summarize(self):
        # summarize information
        x = self.inputs
        summaries = []
        s = x
        for i in range(len(self.dag_weights)):
            s = tf.matmul(s, self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)
        summaries.append(s)
        # global level summary
        for i in range(len(self.global_weights)):
            s = tf.matmul(s, self.global_weights[i])
            s += self.global_bias[i]
            s = self.act_fn(s)
        summaries.append(s)
        return summaries
