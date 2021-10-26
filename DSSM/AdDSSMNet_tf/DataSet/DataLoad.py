import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from collections import namedtuple, OrderedDict
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'share_embed', 'weight_name', 'embed_dim', 'maxlen','dtype'])

# 训练数据集所有特征
feature_columns = [
    SparseFeat(name="uid", voc_size=250000, share_embed=None, embed_dim=256, dtype='string'),
    SparseFeat(name="age", voc_size=100, share_embed=None, embed_dim=8, dtype='string'),
    SparseFeat(name="workid", voc_size=64, share_embed=None, embed_dim=6, dtype='string'),
    SparseFeat(name="height", voc_size=100, share_embed=None, embed_dim=8, dtype='string'),
    SparseFeat(name="sex", voc_size=3, share_embed=None, embed_dim=2, dtype='string'),
    VarLenSparseFeat(name="watched_history_uid_list", voc_size=250000, share_embed='uid', weight_name=None, embed_dim=256, maxlen=100, dtype='string'),
    SparseFeat(name="tuid", voc_size=250000, share_embed="uid", embed_dim=256, dtype='string'),
    SparseFeat(name="tage", voc_size=100, share_embed="age", embed_dim=8, dtype='string'),
    SparseFeat(name="tworkid", voc_size=64, share_embed="workid", embed_dim=6, dtype='string'),
    SparseFeat(name="theight", voc_size=100, share_embed="height", embed_dim=8, dtype='string'),
    SparseFeat(name='tsex', voc_size=3, share_embed="sex", embed_dim=2, dtype='string')
]

# 用户特征及被推荐人特征
user_feature_columns_name = ["uid", "age", "workid", "height", "sex", "watched_history_uid_list"]
item_feature_columns_name = ["tuid", "tage", "tworkid", "theight", "tsex"]
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]

# 定义离散特征集合，离散特征vocabulary
DICT_CATEGORICAL = {
    "uid": [str(i) for i in range(0, 250000)],
    "age": [str(i) for i in range(0, 100)],
    "workid": [str(i) for i in range(0, 64)],
    "height": [str(i) for i in range(0, 100)],
    "sex": [str(i) for i in range(0, 3)]
}


# DEFAULT_VALUES = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]
# COL_NAME = ['uid', 'age', 'workid', 'height', 'sex', 'tuid', 'tage', "tworkid", 'theight', 'tsex', 'watched_history_uid_list']


class DataLoad(tf.keras.layers.Layer):
    def __init__(self, dataPath, DEFAULT_VALUES, COL_NAME, feature_columns):
        self.data_path = dataPath
        self.default_values = DEFAULT_VALUES
        self.feature_columns = feature_columns
        self.col_name = COL_NAME
        self.dataset = self.__init_dataset()
    def _parse_function(self, example_proto):
        item_feats = tf.io.decode_csv(example_proto, record_defaults=self.default_values, field_delim=',', use_quote_delim=False)
        parsed = dict(zip(self.col_name, item_feats))
        feature_dict = {}
        for feat_col in self.feature_columns:
            if isinstance(feat_col, VarLenSparseFeat):
                if feat_col.weight_name is not None:
                    kvpairs = tf.strings.split([parsed[feat_col.name]], '&').values[:feat_col.maxlen]
                    kvpairs = tf.strings.split(kvpairs, ':')
                    kvpairs = kvpairs.to_tensor()
                    feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                    feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
                    feature_dict[feat_col.name] = feat_ids
                    feature_dict[feat_col.weight_name] = feat_vals
                else:
                    feat_ids = tf.strings.split([parsed[feat_col.name]], '&').values[:feat_col.maxlen]
                    feat_ids = tf.reshape(feat_ids, shape=[1, -1])
                    feature_dict[feat_col.name] = feat_ids
            elif isinstance(feat_col, SparseFeat):
                    feature_dict[feat_col.name] = tf.reshape(parsed[feat_col.name], shape=[-1])
            elif isinstance(feat_col, DenseFeat):
                if feat_col.pre_embed is None:
                    feature_dict[feat_col.name] = parsed[feat_col.name]
                elif feat_col.reduce_type is not None:
                    keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                    emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                    emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                # else:
                #     emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                #     feature_dict[feat_col.name] = emb
            else:
                raise Exception("unknown feature_columns....")
        label = tf.reshape(tf.cast(parsed["label"], dtype=tf.int32), [-1])
        return feature_dict, label
    def __init_dataset(self, ):
        # df = pd.read_csv("XXXXX.csv")
        # filenames = tf.io.gfile.glob("/var/chenhaolin/remotedata/dataset/*")
        filenames = tf.io.gfile.glob(self.data_path)
        filenames = tf.data.Dataset.list_files(filenames)
        # dataset = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
        #
        # target = df.pop("label")
        #
        # dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
        #
        # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        dataset = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._parse_function, num_parallel_calls=60)
        # dataset = dataset.batch(64)
        # dataset = dataset.repeat()
        # dataset = dataset.shuffle(buffer_size=64)
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


# 离散多值查找表 转稀疏SparseTensor >> EncodeMultiEmbedding >>tf.nn.embedding_lookup_sparse的sp_ids参数中
# class SparseVocabLayer(tf.keras.layers.Layer):
#
#     def __init__(self, keys, **kwargs):
#         super(SparseVocabLayer, self).__init__(**kwargs)
#         vals = np.arange(1, len(keys)+1)
#         vals = tf.constant(vals, dtype=tf.int32)
#         keys = tf.constant(keys)
#         self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 0)
#
#     def call(self, inputs):
#         input_idx = tf.where(tf.not_equal(inputs, ''))
#         input_sparse = tf.SparseTensor(input_idx, tf.gather_nd(inputs, input_idx), tf.shape(inputs, out_type=tf.int64))
#         return tf.SparseTensor(indices=input_sparse.indices, values=self.table.lookup(input_sparse.values), dense_shape=input_sparse.dense_shape)

class VocabLayer(tf.keras.layers.Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(0, len(keys))
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), -1)
    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (-1)  # mask成 0
            idx = tf.where(masks, idx, paddings)
        return idx
    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


class EmbeddingLookupSparse(tf.keras.layers.Layer):
    def __init__(self, embedding, has_weight=False, **kwargs):
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.embedding = embedding
    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)
    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner="sum")
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner="mean")
        return tf.expand_dims(combiner_embed, 1)
    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight})
        return config


class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding
    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)
    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed
    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config

# 定义model输入特征
def build_input_features(feature_columns):
    input_features = OrderedDict()
    for feat_col in feature_columns:
        if isinstance(feat_col, DenseFeat):
            if feat_col.pre_embed is None:
                input_features[feat_col.name] = tf.keras.layers.Input([1], name=feat_col.name)
            else:
                input_features[feat_col.name] = tf.keras.layers.Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = tf.keras.layers.Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = tf.keras.layers.Input([None], name=feat_col.name, dtype='string')
            if feat_col.weight_name is not None:
                input_features[feat_col.name] = tf.keras.layers.Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))
    return input_features


class DenseToSparseTensor(tf.keras.layers.Layer):
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


def input_from_feature_columns(features, feature_columns, embedding_dict):
    sparse_embedding_list = []
    dense_value_list = []
    for feat_col in feature_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == "string":
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                _inputs_sparse = VocabLayer(keys)(features[feat_col.name])
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                _embed = embedding_dict[feat_col.name](_inputs_sparse)
            else:
                _embed = tf.keras.layers.Embedding(feat_col.voc_size+1, feat_col.embed_dim, embeddings_regularizer=tf.keras.regularizers.l2(0.5), name="Embed_"+feat_col.name)(features[feat_col.name])
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_sparse = DenseToSparseTensor(mask_value=-1)(_inputs_sparse)
            _embed = embedding_dict[feat_col.name](input_sparse)
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))
    return sparse_embedding_list, dense_value_list


def build_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns)
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if "mean" is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          has_weight=True,
                                                                          name='emb_lookup_' + feat_col.name)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          name='emb_lookup_' + feat_col.name)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name], name='emb_lookup_' + feat_col.name)
    return embedding_dict


# 构造自定义矩阵
def build_embedding_matrix(features_columns):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                vocab_size = feat_col.voc_size + 1
                embed_dim = feat_col.embed_dim
                if vocab_name not in embedding_matrix:
                    embedding_matrix[vocab_name] = tf.Variable(
                            initial_value=tf.random.truncated_normal(
                                shape=(vocab_size, embed_dim),
                                mean=0.0,
                                stddev=0.001,
                                dtype=tf.float32
                            ),
                            trainable=True,
                            name=vocab_name + '_embed'
                    )
    return embedding_matrix