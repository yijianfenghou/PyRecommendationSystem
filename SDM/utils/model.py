import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tqdm import tqdm
import random


class DynamicRNN(tf.keras.Model):

    def __init__(self, rnn_cell):
        super(DynamicRNN, self).__init__(self)
        self.cell = rnn_cell

    def call(self, input_data):
        # [batch, time, features] -> [time, batch, features]
        input_data = tf.transpose(input_data, [1, 0, 2])
        outputs = tf.TensorArray(tf.float32, input_data.shape[0])
        state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
        for i in tf.range(input_data.shape[0]):
          output, state = self.cell(input_data[i], state)
          outputs = outputs.write(i, output)
        return tf.transpose(outputs.stack(), [1, 0, 2]), state


class DynamicMultiRNN(Layer):
    def __init__(self, num_units=None, rnn_type='LSTM', return_sequence=True, num_layers=2, num_residual_layers=1, dropout_rate=0.2, forget_bias=1.0, **kwargs):
        self.num_units = num_units
        self.return_sequence = return_sequence
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_residual_layers = num_residual_layers
        self.dropout = dropout_rate
        self.forget_bias = forget_bias
        super(DynamicMultiRNN, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if not self.num_units:
            self.num_units = input_seq_shape.as_list()[-1]
        if self.rnn_type == "LSTM":
            single_cell = tf.keras.layers.LSTMCell(self.num_units, unit_forget_bias=self.forget_bias)
        elif self.rnn_type == "GRU":
            single_cell = tf.keras.layers.GRUCell(self.num_units)
        else:
            raise ValueError("Unknown unit type %s!" % self.rnn_type)
        dropout = self.dropout if tf.keras.backend.learning_phase() == 1 else 0
        single_cell = tf.nn.RNNCellDropoutWrapper(single_cell, input_keep_prob=(1.0 - dropout))
        cell_list = []
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            if residual:
                single_cell_residual = tf.nn.RNNCellResidualWrapper(single_cell)
                cell_list.append(single_cell_residual)
            else:
                cell_list.append(single_cell)

        if len(cell_list) == 1:
            self.final_cell = cell_list[0]
        else:
            self.final_cell = tf.keras.layers.StackedRNNCells(cell_list)
        super(DynamicMultiRNN, self).build(input_shape)

    def call(self, input_list, mask=None, **kwargs):
        rnn_input, sequence_length = input_list

        rnn_output, hidden_state = DynamicRNN(rnn_input)

        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[2])

    def get_config(self):
        config = {
            'num_units': self.num_units,
            'rnn_type': self.rnn_type,
            'return_sequence': self.return_sequence,
            'num_layers': self.num_layers,
            'num_residual_layers': self.num_residual_layers,
            'dropout_rate': self.dropout
        }
        base_config = super(DynamicMultiRNN, self).get_config()
        return dict(base_config.items()+config.items())


class DotAttention(Layer):

    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        super(DotAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `DotAttention` layer should be called on a list of 2 tensors')
        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError('query_size should keep the same dim with key_size')

        super(DotAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, key = inputs
        output = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        if self.scale == True:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def compute_mask(self, inputs, mask=None):
        return mask


class SoftmaxWeightedSum(Layer):

    def __init__(self, dropout_rate=0.2, future_bindding=False, seed=2020, **kwargs):
        self.dropout_rate = dropout_rate
        self.future_binding = future_bindding
        self.seed = seed
        super(SoftmaxWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SoftmaxWeightedSum` layer should be called on a list of 3 tensors')
        if input_shape[0][-1] != input_shape[2][-1]:
            raise ValueError('query_size should keep the same dim with key_mask_size')

        self.drop = Dropout(self.dropout_rate, seed=self.seed)
        super(SoftmaxWeightedSum, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        align, value, key_masks = inputs
        paddings = tf.ones_like(align) * (-2**32 + 1)
        align = tf.where(key_masks, align, paddings)
        if self.future_binding:
            length = value.get_shape().as_list()[1]
            lower_tri = tf.ones([length, length])
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
            masks= tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
            align = tf.where(tf.equal(masks, 0), paddings, align)

        align = tf.nn.softmax(align)
        align = self.dropout(align, training=training)
        output = tf.matmul(align, value)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def get_config(self):
        config = {"dropout_rate": self.dropout_rate, "future_binding": self.future_binding}
        base_config = super(SoftmaxWeightedSum, self).get_config()
        return dict(config.items() + base_config.items())

    def compute_mask(self, inputs, mask=None):
        return mask


class ConcatAttention(Layer):

    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        super(ConcatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ConcatAttention` layer should be called on a list of 2 tensors')
        self.projection_layer = Dense(units=1, activation='tanh')
        super(ConcatAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, key = inputs
        q_k = tf.concat([query, key], axis=-1)
        output = self.projection_layer(q_k)
        if self.scale == True:
            output = output/(key.get_shape().as_list()[-1] ** 0.5)
        output = tf.transpose(output, [0, 2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def compute_mask(self, inputs, mask=None):
        return mask


class AttentionSequencePoolingLayer(Layer):

    def __init__(self):
        pass
