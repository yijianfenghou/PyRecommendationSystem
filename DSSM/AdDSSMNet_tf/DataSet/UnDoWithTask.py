import tensorflow as tf
from collections import namedtuple, OrderedDict
from tensorflow.keras.layers import Layer, Input
import pandas as pd

# 定义输入数据参数类型
# 单值离散型SparseFeat，如topic_id字段
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'embed_dim', 'dtype'])
# 稠密数值类型DenseFeat，如用户访问时间及用户embedding向量等
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
# 多值变长离散特征
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim',
                               'maxlen', 'dtype'])

# 定义离散特征集合 ，离散特征vocabulary
DICT_CATEGORICAL = {
    "uid": [str(i) for i in range(0, 10000)],
    "age": [str(i) for i in range(100)],
    "work_id": [str(i) for i in range(50)],
    "height": [str(i) for i in range(100)],
    "sex": [str(i) for i in range(2)],
    "subeventid": [str(i) for i in range(0, 10000)],
    "match_age": [str(i) for i in range(100)],
    "match_work_location": [str(i) for i in range(50)],
    "match_height": [str(i) for i in range(100)],
    "match_sex": [str(i) for i in range(2)],
}

feature_columns = [
    SparseFeat(name="uid", voc_size=10000, hash_size=None, share_embed=None, embed_dim=100, dtype='string'),
    SparseFeat(name="age", voc_size=100, hash_size=None, share_embed=None, embed_dim=6, dtype='string'),
    SparseFeat(name='work_location', voc_size=50, hash_size=None, share_embed=None, embed_dim=6, dtype='string'),
    SparseFeat(name='height', voc_size=100, hash_size=None, share_embed=None, embed_dim=6, dtype='string'),
    SparseFeat(name='sex', voc_size=2, hash_size=None, share_embed=None, embed_dim=1, dtype='string'),
    SparseFeat(name="subeventid", voc_size=10000, hash_size=None, share_embed="uid", embed_dim=100, dtype='string'),
    SparseFeat(name='match_age', voc_size=100, hash_size=None, share_embed="age", embed_dim=6, dtype='string'),
    SparseFeat(name='match_work_location', voc_size=50, hash_size=None, share_embed="work_id", embed_dim=6, dtype='string'),
    SparseFeat(name='match_height', voc_size=100, hash_size=None, share_embed="height", embed_dim=6, dtype='string'),
    SparseFeat(name='match_sex', voc_size=2, hash_size=None, share_embed="sex", embed_dim=1, dtype='string'),
    # SparseFeat(name='label', voc_size=2, hash_size=None, share_embed=None, embed_dim=1, dtype='int32'),
    # VarLenSparseFeat(name="hist_tuid", voc_size=10000, hash_size=None, share_embed='uid', weight_name=None,
    #                  combiner=None, embed_dim=100, maxlen=40, dtype='string'),
    # DenseFeat(name='client_embed', pre_embed='read_post_id', reduce_type='mean', dim=100, dtype='float32'),
]

# 用户行为序列特征
user_feature_columns_name = ['uid', 'age', 'work_id', 'height']
item_feature_columns_name = ['subeventid', 'match_age', 'match_work_id', 'match_height']
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]


DEFAULT_VALUES = [[''], [0.], [0.0], [0.0], [0.0], [''], [0.0], [0.0], [0.0], [0.0], [0]]
COL_NAME = ['uid', 'age', 'work_location', 'height', 'sex', 'subeventid', 'match_age', 'match_work_location', 'match_height', 'match_sex', 'label']


# def _parse_function(example_proto):
#     item_feats = tf.io.decode_csv(example_proto, DEFAULT_VALUES)
#     # item_feats = example_proto.numpy()
#     parsed = dict(zip(COL_NAME, item_feats))
#
#     feature_dict = {}
#     for feat_col in feature_columns:
#         if isinstance(feat_col, VarLenSparseFeat):
#             if feat_col.weight_name is not None:
#                 kvpairs = tf.strings.split([parsed[feat_col.name]], ",").values[:feat_col.maxlen]
#                 kvpairs = tf.strings.split(kvpairs, ":")
#                 kvpairs = kvpairs.to_tensor()
#                 feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
#                 feat_ids = tf.reshape(feat_ids, shape=[-1])
#                 feat_vals = tf.reshape(feat_vals, shape=[-1])
#                 if feat_col.dtype != 'string':
#                     feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
#                 feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
#                 feature_dict[feat_col.name] = feat_ids
#                 feature_dict[feat_col.weight_name] = feat_vals
#             else:
#                 feat_ids = tf.strings.split([parsed[feat_col.name]], ",").values[:feat_col.maxlen]
#                 feat_ids = tf.reshape(feat_ids, shape=[-1])
#                 if feat_col.dtype != 'string':
#                     feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
#                 feature_dict[feat_col.name] = feat_ids
#         elif isinstance(feat_col, SparseFeat):
#             feature_dict[feat_col.name] = tf.convert_to_tensor(parsed[feat_col.name])
#         elif isinstance(feat_col, DenseFeat):
#             feature_dict[feat_col.name] = tf.convert_to_tensor(parsed[feat_col.name])
#         else:
#             raise Exception("unknown feature_columns....")
#
#     label = tf.convert_to_tensor(parsed['label'])
#     # return label
#     return (feature_dict, label)


pad_shapes = {}
pad_values = {}

for feat_col in feature_columns:
    if isinstance(feat_col, VarLenSparseFeat):
        max_tokens = feat_col.maxlen
        pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
        pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
        if feat_col.weight_name is not None:
            pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)
    elif isinstance(feat_col, SparseFeat):
        if feat_col.dtype == 'string':
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = '0'
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0
    elif isinstance(feat_col, DenseFeat):
        if not feat_col.pre_embed:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
            pad_values[feat_col.name] = 0

pad_shapes = (pad_shapes, (tf.TensorShape([])))
pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

# 文件路径
path1 = "C:/Users/EDZ/Desktop/result1.csv"

filenames = tf.data.Dataset.list_files([path1, ])

# 处理pandas数据
df = pd.read_csv(path1).head()
labels = df.pop("label")
dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

# dataset = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 1024
# line_dataset = dataset.map(lambda row: tf.py_function(_parse_function, [row], Tout=[tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]))
#
# for line in line_dataset:
#     print(line)

# dataset = dataset.map(_parse_function, num_parallel_calls=60)
# dataset = dataset.map(lambda row: tf.py_function(_parse_function, [row], Tout=[tf.string]))

# for line in dataset:
#     print(line)

dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓存区中随机打乱数据
dataset = dataset.repeat()
dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=pad_shapes)  # 每1024条数据为一个batch，生成一个新的Datasets
# Dataset.prefetch() 开启预加载数据，使得在 GPU 训练的同时 CPU 可以准备数据
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 多值查找表稀疏SparseTensor >> EncodeMultiEmbedding >> tf.nn.embedding_lookup_sparse的sp_ids参数中
class SparseVocabLayer(Layer):

    def __init__(self, keys, mask_value=None, **kwargs):
        super(SparseVocabLayer, self).__init__(**kwargs)
        vals = tf.range(1, len(keys) + 1)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def call(self, inputs):
        input_idx = tf.where(tf.not_equal(inputs, ""))
        input_sparse = tf.SparseTensor(input_idx, tf.gather_nd(inputs, input_idx), tf.shape(inputs, out_type=tf.int64))
        return tf.SparseTensor(indices=input_sparse.indices, values=self.table.lookup(input_sparse.values), dense_shape=input_sparse.dense_shape)


# 自定义Embedding层，初始化时，需要传入预先定义好的embedding矩阵，好处可以共享embedding矩阵
class EncodeMultiEmbedding(Layer):

    def __init__(self, embedding, has_weight=False, **kwargs):
        super(EncodeMultiEmbedding, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.embedding = embedding

    def build(self, input_shape):
        super(EncodeMultiEmbedding, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner='sum')
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner='mean')
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EncodeMultiEmbedding, self).get_config()
        config.update({"has_weight": self.has_weight})
        return config


# 稠密权重转稀疏格式输入到tf.nn.embedding_lookup_sparse的sp_weights参数中
class Dense2SparseTensor(Layer):

    def __init__(self):
        super(Dense2SparseTensor, self).__init__()

    def call(self, dense_tensor):
        weight_idx = tf.where(tf.not_equal(dense_tensor, tf.constant(-1, dtype=tf.float32)))
        weight_sparse = tf.SparseTensor(weight_idx, tf.gather_nd(dense_tensor, weight_idx), tf.shape(dense_tensor, out_type=tf.int64))
        return weight_sparse

    def get_config(self):
        config = super(Dense2SparseTensor, self).get_config()
        return config


# 自定义dnese层含BN， dropout
class CustomDense(Layer):

    def __init__(self, units=32, activation='tanh', dropout_rate =0, use_bn=False, seed=1024, tag_name="dnn", **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.tag_name = tag_name

    # build方法一般定义Layer需要被训练的参数
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True,
                                      name='kernel_' + self.tag_name)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='random_normal',
                                    trainable=True,
                                    name='bias_' + self.tag_name)

        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()

        self.dropout_layers = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(self.activation, name=self.activation + '_' + self.tag_name)

        super(CustomDense, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs, training=None, **kwargs):
        fc = tf.matmul(inputs, self.weight) + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        out_fc = self.activation_layers(fc)

        return out_fc

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法，保存模型不写这部分会报错
    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_bn': self.use_bn,
                       'dropout_rate': self.dropout_rate, 'seed': self.seed, 'name': self.tag_name})
        return config


# cos 相似度计算层
class Similarity(Layer):

    def __init__(self, gamma=1, axis=-1, type_sim='cos', **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type_sim = type_sim
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        if self.type_sim == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)
        cosine_score = tf.reduce_sum(tf.multiply(query, candidate), -1)
        cosine_score = tf.divide(cosine_score, query_norm * candidate_norm + 1e-8)
        cosine_score = tf.clip_by_value(cosine_score, -1, 1.0) * self.gamma
        return tf.expand_dims(cosine_score, 1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


# 自定损失函数，加权交叉熵损失
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, pos_weight=1.2, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:, None]
        ce = ce * (1 - y_true) + self.pos_weight * ce * (y_true)

        return ce

    def get_config(self, ):
        config = {'pos_weight': self.pos_weight, 'from_logits': self.from_logits, 'name': self.name}
        base_config = super(WeightedBinaryCrossEntropy, self).get_config()
        return base_config.uptate(config)


# 定义model输入特征
def build_input_features(features_columns, prefix=""):
    input_features = OrderedDict()
    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            if feat_col.pre_embed is None:
                input_features[feat_col.name] = Input([1], name=feat_col.name)
            else:
                input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            else:
                input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype='string')
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构造自定义embedding层matrix
def build_embedding_matrix(features_columns):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                vocab_size = feat_col.voc_size
                embed_dim = feat_col.embed_dim
                if vocab_name not in embedding_matrix:
                    embedding_matrix[vocab_name] = tf.Variable(initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0, stddev=0.0, dtype=tf.float32), trainable=True, name=vocab_name+"_embed")
    return embedding_matrix


# 构造自定义embedding层
def build_embedding_dict(features_columns, embedding_matrix):
    embedding_dict = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name], name="EncodeMuiltiEmb_"+feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.weight_name is not None:
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name], has_weight=True, name="EncodeMultiEmb_"+feat_col.name)
            else:
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name], name="EncodeMultiEmb_"+feat_col.name)
    return embedding_dict


# dense与embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict):
    sparse_embedding_list = []
    dense_embedding_list = []
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                _input_sparse = SparseVocabLayer(keys)(features[feat_col.name])

        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                _embed = embedding_dict[feat_col.name](_input_sparse)
            else:
                _embed = tf.keras.layers.Embedding(feat_col.voc_size+1, feat_col.embed_dim, embeddings_regularizer=tf.keras.regularizers.l2(0.5), name='Embed+'+feat_col.name)(features[feat_col.name])
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                _weight_sparse = Dense2SparseTensor()(features[feat_col.weight_name])
                _embed = embedding_dict[feat_col.name]([_input_sparse, _weight_sparse])
            else:
                _embed = embedding_dict[feat_col.name](_input_sparse)
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, DenseFeat):
            dense_embedding_list.append(features[feat_col.name])
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_embedding_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = tf.keras.layers.Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = tf.keras.layers.Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return tf.keras.layers.Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return tf.keras.layers.Flatten()(concat_func(dense_value_list))
    else:
        raise Exception("dnn_feature_columns can not be empty list")


if __name__ == "__main__":
    feature_columns = user_feature_columns + item_feature_columns
    # 构建embedding_dict
    embedding_matrix = build_embedding_matrix(feature_columns)
    embedding_dict = build_embedding_dict(feature_columns, embedding_matrix)

    # user特征处理
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns, embedding_dict)
    # print(user_sparse_embedding_list)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # item 特征处理
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features, item_feature_columns, embedding_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    # user tower