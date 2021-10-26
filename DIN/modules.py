import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense


class Attention_Layer(Layer):

    def __init__(self, att_hidden_units, activation='PReLU'):
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(units=unit, activation=tf.keras.layers.PReLU()) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        # (None, seq_len*d*2)
        q = tf.tile(q, multiples=[1, k.shape[1]])
        # (None, seq_len, d*2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])

        # q, k, out product should concat
        info = tf.concat([q, k, q-k, q*k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        # (None, seq_len, 1)
        outputs = self.att_final_dense(info)
        # (None, seq_len)
        outputs = self.squeeze(outputs, axis=-1)

        # (None, seq_len)
        padding = tf.ones_like(outputs) * (-2**32 + 1)
        # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), padding, outputs)

        # softmax
        # (None, seq_len)
        outputs = tf.nn.softmax(logits=outputs)
        # (None, 1, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)

        # (None, 1, d*2)
        outputs = tf.matmul(outputs, v)
        outputs = tf.squeeze(outputs, axis=1)
        return outputs


class Dice(Layer):

    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, inputs):
        inputs_normed = self.bn(inputs)
        inputs_p = tf.sigmoid(inputs_normed)
        return self.alpha * (1.0 - inputs_p) * inputs + inputs_p * inputs