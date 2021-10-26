import itertools

import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dropout, Dense


class DynamicMultiRNN(Layer):
    def __init__(self, num_units=None, rnn_type='LSTM', return_sequence=True, num_layers=2, num_residual_layers=1, dropout_rate=0.2, forget_bias=1.0, **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_residual_layers = num_residual_layers
        self.dropout = dropout_rate
        self.forget_bias = forget_bias
        super(DynamicMultiRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        # tf.contrib.rnn.BasicLSTMCell 改为 tf.keras.layers.LSTMCell
        # tf.contrib.rnn.MultiRNNCell 改为 tf.keras.layers.StackedRNNCells
        # tf.contrib.rnn.static_rnn 改为 keras.layers.RNN(cell, unroll=True)

        if self.rnn_type == "LSTM":
            single_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
        elif self.rnn_type == "GRU":
            single_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units)
        else:
            raise ValueError("Unknown unit type %s!" % self.rnn_type)
        dropout = self.dropout if tf.keras.backend.learning_phase() == 1 else 0
        single_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        cell_list = []
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            if residual:
                single_cell_residual = tf.compat.v1.nn.rnn_cell.ResidualWrapper(single_cell)
                cell_list.append(single_cell_residual)
            else:
                cell_list.append(single_cell)
        if len(cell_list) == 1:
            self.final_cell = cell_list[0]
        else:
            self.final_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)
        super(DynamicMultiRNN, self).build(input_shape)

    def call(self, input_list, mask=None, training=None):
        rnn_input, sequence_length = input_list
        with tf.name_scope("rnn"), tf.compat.v1.variable_scope("rnn", reuse=tf.compat.v1.AUTO_REUSE):
            rnn_output, hidden_state = tf.compat.v1.nn.dynamic_rnn(self.final_cell, inputs=rnn_input, sequence_length=tf.squeeze(sequence_length), dtype=tf.float32, scope=self.name)
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

    def get_config(self, ):
        config = {
            'num_units': self.num_units,
            'rnn_type': self.rnn_type,
            'return_sequence': self.return_sequence,
            'num_layers': self.num_layers,
            'num_residual_layers': self.num_residual_layers,
            'dropout_rate': self.dropout
        }
        base_config = super(DynamicMultiRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotAttention(Layer):
    """
    :param query: [batch_size, 1, C]
    :param key:   [batch_size, T, C]
    :return:      [batch_size, 1, T]
    """
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

    def compute_mask(self, inputs, mask):
        return mask


class SoftmaxWeightedSum(Layer):
    """
    :param align:           [batch_size, 1, T]
    :param value:           [batch_size, T, units]
    :param key_masks:       [batch_size, 1, T]
                            2nd dim size with align
    :param drop_out:
    :param future_binding:
    :return:                weighted sum vector
                            [batch_size, 1, units]
    """
    def __init__(self, dropout_rate=0.2, future_binding=False, seed=2020, **kwargs):
        self.dropout_rate = dropout_rate
        self.future_binding = future_binding
        self.seed = seed
        super(SoftmaxWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SoftmaxWeightedSum` layer should be called '
                             'on a list of 3 tensors')
        if input_shape[0][-1] != input_shape[2][-1]:
            raise ValueError('query_size should keep the same dim with key_mask_size')
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        super(SoftmaxWeightedSum, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        align, value, key_masks = inputs
        paddings = tf.ones_like(align) * (-2 ** 32 + 1)
        align = tf.where(key_masks, align, paddings)
        if self.future_binding:
            length = value.get_shape().as_list()[1]
            lower_tri = tf.ones([length, length])
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
            masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
            align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        align = self.dropout(align, training=training)
        output = tf.matmul(align, value)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate, 'future_binding': self.future_binding}
        base_config = super(SoftmaxWeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask


class ConcatAttention(Layer):
    """
    :param query: [batch_size, T, C_q]
    :param key:   [batch_size, T, C_k]
    :return:      [batch_size, 1, T]
        query_size should keep the same dim with key_size
    """
    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        super(ConcatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ConcatAttention` layer should be called on a list of 2 tensors')
        self.projection_layer = Dense(units=1, activation='tanh')
        super(ConcatAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, key = inputs
        q_k = tf.concat([query, key], axis=-1)
        output = self.projection_layer(q_k)
        if self.scale == True:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        output = tf.transpose(output, [0, 2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def compute_mask(self, inputs, mask):
        return mask


class AttentionSequencePoolingLayer(Layer):
    """
    :param query:           [batch_size, 1, C_q]
    :param keys:            [batch_size, T, C_k]
    :param keys_length:      [batch_size, 1]
    :return:                [batch_size, 1, C_k]
    """
    def __init__(self, dropout_rate=0, **kwargs):
        self.dropout_rate = dropout_rate
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SequenceFeatureMask` layer should be called on a list of 3 inputs')
        self.concat_att = ConcatAttention()
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=False)
        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        queries, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        queries = tf.tile(queries, [1, hist_len, 1])  # [batch_size, T, units]
        attention_score = self.concat_att([queries, keys])  # [batch_size, 1, units]

        outputs = self.softmax_weight_sum([attention_score, keys, key_masks])
        # [batch_size, units]
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask


class SelfMultiHeadAttention(Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, T, C]
    """
    def __init__(self, num_units=8, head_num=4, scale=True, dropout_rate=0.2, future_binding=True, use_layer_norm=True, use_res=True,
                 seed=2020, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.num_units = num_units
        self.head_num = head_num
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.future_binding = future_binding
        self.use_layer_norm = use_layer_norm
        self.use_res = use_res
        self.seed = seed
        super(SelfMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `SelfMultiHeadAttention` layer should be called on a list of 2 tensors')
        if len(input_shape[0]) != 3 or len(input_shape[1]) != 2:
            raise ValueError('input: [N, T_k, d_model], key masks: [N, key_seqlen]')
        embedding_size = int(input_shape[0][-1])
        if self.num_units == None:
            self.num_units = embedding_size
        self.W = self.add_weight(
            name='Q_K_V',
            shape=[embedding_size, self.num_units*3],
            dtype=tf.float32,
            initializer=tf.keras.initializers.truncated_normal(seed=self.seed)
        )
        self.W_output = self.add_weight(
            name='output_W',
            shape=[self.num_units, self.num_units],
            dtype=tf.float32,
            initializer=tf.keras.initializers.truncated_normal(seed=self.seed)
        )

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=self.future_binding, seed=self.seed)
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.seq_len_max = int(input_shape[0][1])
        # Be sure to call this somewhere!
        super(SelfMultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        input_info, keys_length = inputs

        hist_len = input_info.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        key_masks = tf.squeeze(key_masks, axis=1)

        Q_K_V = tf.tensordot(input_info, self.W, axes=(-1, 0))  # [N T_q D*3]
        querys, keys, values = tf.split(Q_K_V, 3, -1)

        # head_num None F D
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # (h*N, T_q, C/h)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)      # (h*N, T_k, C/h)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)  # (h*N, T_k, C/h)

        # (h*N, T_q, T_k)
        align = self.attention([querys, keys])

        key_masks = tf.tile(key_masks, [self.head_num, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input_info)[1], 1])  # (h*N, T_q, T_k)

        outputs = self.softmax_weight_sum([align, values, key_masks])               # (h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, self.head_num, axis=0), axis=2)       # (N, T_q, C)

        outputs = tf.tensordot(outputs, self.W_output, axes=(-1, 0))    # (N, T_q, C)
        outputs = self.dropout(outputs, training=training)
        if self.use_res:
            outputs += input_info
        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.num_units)

    def get_config(self, ):
        config = {
            'num_units': self.num_units,
            'head_num': self.head_num,
            'scale': self.scale,
            'dropout_rate': self.dropout_rate,
            'future_binding': self.future_binding,
            'use_layer_norm': self.use_layer_norm,
            'use_res': self.use_res,
            'seed': self.seed
        }
        base_config = super(SelfMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask


class UserAttention(Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param keys: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, 1, C]
    """
    def __init__(self, num_units=None, activation='tanh', use_res=True, dropout_rate=0, scale=True, seed=2020, **kwargs):
        self.scale = scale
        self.num_units = num_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.use_res = use_res
        super(UserAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `UserAttention` layer should be called on a list of 3 tensors')
        if self.num_units == None:
            self.num_units = input_shape[0][-1]
        self.dense = Dense(self.num_units, activation=self.activation)
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, seed=self.seed)
        super(UserAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        user_query, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        query = self.dense(user_query)

        align = self.attention([query, keys])

        output = self.softmax_weight_sum([align, keys, key_masks])

        if self.use_res:
            output += keys
        return tf.math.reduce_mean(output, 1, keep_dims=True)

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][2])

    def compute_mask(self, inputs, mask):
        return mask


class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(
            shape=[self.size],
            initializer=tf.keras.initializers.Zeros,
            dtype=tf.float32,
            trainable=False,
            name="bias"
        )
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx

        loss = tf.nn.sampled_softmax_loss(
            weights=embeddings,  # self.item_embedding.
            biases=self.zero_bias,
            labels=label_idx,
            inputs=inputs,
            num_sampled=self.num_sampled,
            num_classes=self.size,  # self.target_song_size
        )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


gate_input = concat_func([prefer_output, short_output, user_emb_output])
gate = Dense(units, activation='sigmoid')(gate_input)

gate_output = Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1-x[0], x[2]))([gate, short_output, prefer_output])
gate_output_reshape = Lambda(lambda x: tf.squeeze(x, 1))(gate_output)
# gate_output_reshape即输出的user向量




