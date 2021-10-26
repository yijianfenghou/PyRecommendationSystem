from copy import copy

from tensorflow.keras.layers import Layer, Flatten, Lambda
from tensorflow.keras.initializers import glorot_normal, Zeros
import tensorflow as tf
from tensorflow.keras import backend as K
import itertools
from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'share_embed', 'weight_name', 'embed_dim', 'maxlen','dtype'])

class Linear(Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):
        self.l2_reg = l2_reg
        if mode not in  [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")

        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.use_bias = self.add_weight(name="linear_bias", shape=(1, ), initializer=Zeros(), trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(name="linear_kernel", shape=(int(input_shape[-1]), 1), initializer=glorot_normal(self.seed), regularizer=l2(self.l2_reg), trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(name="linear_kernel", shape=(int(input_shape[1][-1]), 1), initializer=glorot_normal(self.seed), regularizer=l2(self.l2_reg), trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            spase_input = inputs
            linear_logit = reduce_sum(spase_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.use_bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(Linear, self).get_config()
        base_config.update(config)
        return base_config


class NoMask(Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask=None):
        return mask


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        raise tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_sum(input_tensor, axis=None, keep_dims=None, name=name, reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keep_dims, name=name, reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def reduce_mean(input_tensor, axis=None, keep_dims=None, name=name, reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keep_dims, name=name, reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True, support_dense=True, support_group=False):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding


def get_linear_logit(features, feature_columns, units=1, use_bais=False, seed=1024, prefix='linear', l2_reg=0, sparse_feat_refine_weight=None):
    linear_feature_columns = copy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1, embedding_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1, embedding_initializer=Zeros())

    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix+str(i)[0]) for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):
        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0]*tf.expand_dims(x[1], axis=1))([sparse_input, dense_input])
            linear_logit = Lin


class Add(Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.Add(inputs)


def combined_dnn_inputs(sparse_embedding_list, dense_embedding_list):
    if len(sparse_embedding_list) > 0 and len(dense_embedding_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_embedding_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_embedding_list) > 0:
        return Flatten()(concat_func(dense_embedding_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


class SENETLayer(Layer):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio
        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A AttentionalFM layer should be called on a list of at least 2 inputs')

        self.field_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.field_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(self.field_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(reduction_size, self.field_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = tf.keras.layers.Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        inputs = concat_func(inputs, axis=-1)
        Z = reduce_mean(inputs)

        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))

        return tf.split(V, self.field_size, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return [None] * self.field_size

    def get_config(self,):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SENETLayer, self).get_config()
        base_config.update(config)
        return base_config


class BilinearInteraction(Layer):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``. Its length is ``filed_size``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2,embedding_size)``.
      Arguments
        - **bilinear_type** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """
    def __init__(self, bilinear_type="interaction", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])
                if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(seed=self.seed), name='bilinear_weight')
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(seed=self.seed), name='bilinear_weight'+str(i)) for i in range(len(input_shape)-1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(seed=self.seed), name='bilinear_weight'+str(i)+"_"+str(j)) for i,j in itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError
        super(BilinearInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        n = len(inputs)
        if self.bilinear_type == "all":
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]
            p = [tf.multiply(vidots[i], inputs[j]) for i,j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n-1)]
            p = [tf.multiply(vidots[i], inputs[j]) for i,j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1]) for v,w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
        output = concat_func(p, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        field_size = len(input_shape)
        embedding_size = input_shape[0][-1]

        return (None, field_size * (field_size - 1) // 2, embedding_size)

    def get_config(self,):
        config = {'bilinear_type': self.bilinear_type, 'seed': self.seed}
        base_config = super(BilinearInteraction, self).get_config()
        base_config.update(config)
        return base_config