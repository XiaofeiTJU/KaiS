import numpy as np
import tensorflow as tf


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


class GraphCNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 max_depth, act_fn, scope='gcn'):

        self.inputs = inputs
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn
        self.scope = scope

        # initialize message passing transformation parameters
        self.prep_weights, self.prep_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.proc_weights, self.proc_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        self.agg_weights, self.agg_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        self.outputs = self.forward()

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

    def forward(self):
        x = self.inputs
        # Raise x into higher dimension
        for l in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[l])
            x += self.prep_bias[l]
            x = self.act_fn(x)
        for d in range(self.max_depth):
            y = x
            # Process the features
            for l in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[l])
                y += self.proc_bias[l]
                y = self.act_fn(y)
            # Aggregate features
            for l in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = self.act_fn(y)

            # assemble neighboring information
            x = x + y
        return x
