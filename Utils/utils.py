from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

SparseFeat = namedtuple("SparseFeat", ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'embed_dim', 'dtype'])
Densefeat = namedtuple("DenseFeat", ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple("VarLenSparseFeat", ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'weight_name', 'combiner', 'embed_dim', 'maxlen', 'dtype'])


# 离散特征查找表
class VocabLayer(Layer):

    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys)+2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs, **kwargs):
        index = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            padding = tf.ones_like(index)*(-1)
            index = tf.where(masks, index, padding)
        return index

    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value,})
        return config


# multi-hot特征EmbeddingLook层
class EmbeddingLookupSparse(Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):
        super(EmbeddingLookupSparse, self).__init__()
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding

    def call(self, inputs, **kwargs):
        if self.has_weight:
            index, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=index, sp_weights=val, combiner=self.combiner)
        else:
            index = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=index, sp_weights=None, combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner': self.combiner})
        return config


# 单值离散值特征EmbeddingLookup层
class EmbeddingLookup(Layer):

    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def call(self, inputs, **kwargs):
        index = tf.cast(inputs, tf.int32)
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=index)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config


# 稠密转稀疏
class DenseToSparseTensor(Layer):

    def __init__(self, mask_value=-1, **kwargs):
        self.mask_value = mask_value
        super(DenseToSparseTensor, self).__init__()

    def call(self, inputs, **kwargs):
        index = tf.where(tf.not_equal(inputs, tf.constant(self.mask_value, dtype=inputs.dtype)))
        sparse_tensor = tf.SparseTensor(index, tf.gather_nd(inputs, index), tf.shape(inputs, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({"mask_value": self.mask_value})
        return config


# 自定义hash层
class HashLayer(Layer):
    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        super(HashLayer, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero

    def call(self, inputs, mask=None, **kwargs):
        if inputs.dtype != tf.string:
            zero = tf.as_string(tf.ones([1], dtype=inputs.dtype)*(-1))
            inputs = tf.as_string(inputs,)
        else:
            zero = tf.as_string(tf.ones([1], dtype=inputs.dtype)*(-1))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets-1
        hash_x = tf.strings.to_hash_bucket_fast(inputs, num_buckets, name=None)
        if self.mask_zero:
            masks = tf.not_equal(inputs, zero)
            paddings = tf.ones_like(hash_x)*(-1)
            hash_x = tf.where(masks, hash_x, paddings)
        return hash_x

    def get_config(self, ):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, })
        return config


class PredictionLayer(Layer):
    def __init__(self, task='binary', **kwargs):
        if task in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        super(PredictionLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.task == 'binary':
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))
        return output

    def get_config(self):
        config = {'task': self.task}
        base_config = super(PredictionLayer, self).get_config()
        return dict(**base_config, **config)








