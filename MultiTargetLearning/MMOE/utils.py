from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from collections import namedtuple

SpareFeat = namedtuple("SpareFeat", ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'weight_name', 'combiner', 'embed_dim', 'maxlen', 'dtype'])


class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', use_bias=True, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, weight_initializer='VarianceScaling',
                 weight_regularizer=None, weight_constraint=None,
                 activity_regularizer=None, seed=1024, **kwargs):

        # Weight parameter
        self.weights = None
        self.weight_initializer = tf.keras.initializers.get(weight_initializer)
        self.weight_regularizer = tf.keras.regularizers.get(weight_regularizer)
        self.weight_constraint = tf.keras.constraints.get(weight_constraint)

        # Activation parameter
        self.activation = activation

        # Bias parameter
        self.bias = None
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        # hidden_units parameter
        self.hidden_units = hidden_units
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn_layer = [Dense(units=self.hidden_units[i],
                                activation=self.activation,
                                use_bias=self.use_bias,
                                weight_initializer=self.weight_initializer,
                                bias_initializer=self.bias_initializer,
                                weight_regularizer=self.weight_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                activity_regularizer=self.activity_regularizer,
                                weight_constraint=self.weight_constraint,
                                bias_constraint=self.bias_constraint,
                                ) for i in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        dnn_input = inputs
        dnn_out = None

        # DNN part
        for i in range(len(self.hidden_units)):
            if i == len(self.hidden_units) - 1:
                dnn_out = self.dnn_layer[i](dnn_input)

            dnn_input = self.dnn_layer[i](dnn_input)

        return dnn_out

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units, 'seed': self.seed,
                  'use_bias': self.use_bias, 'weight_initializer': self.weight_initializer,
                  'bias_initializer': self.bias_initializer, 'weight_regularizer': self.weight_regularizer,
                  'bias_regularizer': self.bias_regularizer, 'activity_regularizer': self.activity_regularizer,
                  'weight_constraint': self.weight_constraint, 'bias_constraint': self.bias_constraint, }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# 离散特征查找表层
class VocabLayer(Layer):

    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__()
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs, **kwargs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (-1)
            idx = tf.where(masks, idx, paddings)
        return idx

    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


# multi-hot 特征EmbeddingLookup层
class EmbeddingLookupSparse(Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):
        super(EmbeddingLookupSparse, self).__init__()
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding

    def call(self, inputs, **kwargs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner=self.combiner)

        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner': self.combiner})
        return config


# 单值离散特征EmbeddingLookup层
class EmbeddingLookup(Layer):

    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__()
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs, **kwargs):
        idx = tf.cast(inputs, tf.int32)
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config


# 稠密转稀疏
class DenseToSparseTensor(Layer):

    def __init__(self, mask_value=-1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value

    def call(self, dense_tensor):
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value, dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config


# 自定义hash层
class HashLayer(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(HashLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            zero = tf.as_string(tf.ones([1], dtype=x.dtype) * (-1))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.ones([1], dtype=x.dtype) * (-1))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)
        if self.mask_zero:
            #             mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            masks = tf.not_equal(x, zero)
            paddings = tf.ones_like(hash_x) * (-1)  # mask成 -1
            hash_x = tf.where(masks, hash_x, paddings)
        #             hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, })
        return config


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    """
    def __init__(self, task='binary', **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MMoELayer(Layer):

    def __init__(self, units_experts, num_experts, num_tasks, use_expert_bias=True, use_gate_bias=True,
                 expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros',
                 expert_bias_regularizer=None, gate_bias_regularizer=None,
                 expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_weight_initializer='VarianceScaling', gate_weight_initializer='VarianceScaling',
                 expert_weight_regularizer=None, gate_weight_regularizer=None,
                 expert_weight_constraint=None, gate_weight_constraint=None,
                 activity_regularizer=None, **kwargs):
        super(MMoELayer, self).__init__(**kwargs)

        self.units_experts = units_experts
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_weights = None
        self.gate_weights = None
        self.expert_weight_initializer = tf.keras.initializers.get(expert_weight_initializer)
        self.gate_weight_initializer = tf.keras.initializers.get(gate_weight_initializer)
        self.expert_weight_regularizer = tf.keras.regularizers.get(expert_weight_regularizer)
        self.gate_weight_regularizer = tf.keras.regularizers.get(gate_weight_regularizer)
        self.expert_weight_constraint = tf.keras.constraints.get(expert_weight_constraint)
        self.gate_weight_constraint = tf.keras.constraints.get(gate_weight_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(
                tf.keras.layers.Dense(
                    self.units_experts,
                    activation=self.expert_activation,
                    use_bias=self.use_expert_bias,
                    kernel_initializer=self.expert_weight_initializer,
                    bias_initializer=self.expert_bias_initializer,
                    kernel_regularizer=self.expert_weight_regularizer,
                    bias_regularizer=self.expert_bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.expert_weight_constraint,
                    bias_constraint=self.expert_bias_constraint,
                    name='expert_net_{}'.format(i)
                )
            )

        for i in range(self.num_tasks):
            self.gate_layers.append(
                tf.keras.layers.Dense(
                    self.num_experts,
                    activation=self.gate_activation,
                    use_bias=self.use_gate_bias,
                    kernel_initializer=self.gate_weight_initializer,
                    bias_initializer=self.gate_bias_initializer,
                    kernel_regularizer=self.gate_weight_regularizer,
                    bias_regularizer=self.gate_bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.gate_weight_constraint,
                    bias_constraint=self.gate_bias_constraint,
                    name='gate_net_{}'.format(i)
                )
            )

        def call(self, inputs, **kwargs):
            expert_outputs, gate_outputs, final_outputs = [], [], []
            # inputs: (batch_size, embedding_size)
            for expert_layer in self.expert_layers:
                expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
                expert_outputs.append(expert_output)

            # batch_size * units * num_experts
            expert_outputs = tf.concat(expert_outputs, 2)

            # [(batch_size, num_experts), ....]
            for gate_layer in self.gate_layers:
                gate_outputs.append(gate_layer(inputs))

            for gate_output in gate_outputs:
                # (batch_size, 1, num_experts)
                expanded_gate_output = tf.expand_dims(gate_output, axis=1)

                # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
                weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output, self.units_experts, axis=1)

                # (batch_size, units)
                final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

            # [(batch_size, units), ....]  size: num_task
            return final_outputs