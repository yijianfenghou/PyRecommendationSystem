import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 1000000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

xs = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
ys = tf.placeholder(tf.float32, [None, 10])

weights = {
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X(128, 28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, X_in, init_state=init_state, time_major=False)
