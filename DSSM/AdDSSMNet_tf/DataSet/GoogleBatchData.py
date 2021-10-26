import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, initializers


class HashEmbedding(Layer):

    def __init__(self, hash_bucket_size, embedding_dim, regularizer=0.5, initializer='uniform', trainable=False,
                 **kwargs):

        super(HashEmbedding, self).__init__(**kwargs)
        self._hash_bucket_size = hash_bucket_size
        self._embedding_dim = embedding_dim
        self._regularizer = regularizers.l2(regularizer)
        self._initializer = initializers.get(initializer)
        self._trainable = trainable

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(self._hash_bucket_size, self._embedding_dim),
            regularizer=self._regularizer,
            initializer=self._initializer,
            trainable=self._trainable,
            name="embedding"
        )
        super(HashEmbedding, self).build(input_shape)

    def call(self, inputs, mean=False, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        outputs = K.dot(inputs, self.embedding)
        if mean is True:
            outputs = tf.math.divide_no_nan(
                outputs,
                tf.tile(tf.reduce_sum(inputs, axis=-1, keepdims=True), (1, self._embedding_dim))
            )
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._embedding_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hash_bucket_size': self._hash_bucket_size,
            'embedding_dim': self._embedding_dim,
            'regularizer': self._initializer,
            'trainanle': self._trainable
        })
        return config


class L2Normalization(Layer):

    def __init__(self, kernel_size, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self._weights = self.add_weight(
            shape=(self.kernel_size,),
            initializer=initializers.glorot_normal,
            trainable=True,
            name="l2norm_weights"
        )

        self._bais = self.add_weight(
            shape=(self.kernel_size,),
            initializer=initializers.zeros,
            trainable=True,
            name='l2norm_bais'
        )

    def call(self, inputs, **kwargs):
        outputs = self._weights * inputs + self._bais
        return K.l2_normalize(outputs)


def build_model():
    # features = ["uid", "age", "height", "workid", "sex"]
    #
    # # 连续值特征
    # for feature in ['age', 'height', 'workid']:
    #     feature = tf.feature_column.numeric_column(feature)
    #     features.append(feature)

    # 数值型特征的处理方式
    seed_numerical_inputs = {}
    cand_numerical_inputs = {}
    seed_numerical_features = {}
    cand_numerical_features = {}

    # 连续值特征转换为离散/分桶区间
    age_numeric = tf.feature_column.numeric_comn('age')
    age_bucket = tf.feature_column.bucketized_column(age_numeric, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    age_bucket_dense = tf.keras.layers.DenseFeatures(age_bucket, trainable=False, name='age')
    seed_age_input = tf.keras.layers.Input(shape=(1,), name='seed_age_input')
    cand_age_input = tf.keras.layers.Input(shape=(1,), name='cand_age_input')
    seed_numerical_inputs['seed_age'] = seed_age_input
    cand_numerical_inputs['cand_age'] = cand_age_input
    seed_numerical_features['seed_age'] = age_bucket_dense({"age": seed_age_input})
    cand_numerical_features['cand_age'] = age_bucket_dense({"age": cand_age_input})

    # # 性别类别(非数值型)转化onehot
    # sex_category = tf.feature_column.categorical_column_with_vocabulary_list('age', ['f', 'm', 'unknown'])
    # sex_onehot = tf.feature_column.indicator_column(sex_category)
    # # 性别类别(非数值型存储)转化为embedding
    # sex_embedding = tf.feature_column.embedding_column(sex_onehot, dimension=2)

    # 处理uid类别值情况
    user_id = tf.feature_column.categorical_column_with_hash_bucket(key='user_id', hash_bucket_size=100000,
                                                                         dtype=tf.string)
    user_id_onehot = tf.feature_column.indicator_column(user_id)
    # 变成稠密特征
    user_id_dense = tf.keras.layers.DenseFeatures(user_id_onehot, trainable=False, name='user_id')
    seed_user_id_input = tf.keras.layers.Input(shape=(1,), name='seed_user_id', dtype=tf.string)
    cand_user_id_input = tf.keras.layers.Input(shape=(1,), name='cand_user_id', dtype=tf.string)
    seed_user_id = user_id_dense({"user_id": seed_user_id_input})
    cand_user_id = user_id_dense({"user_id": cand_user_id_input})

    for feat in ["height", "workid"]:
        feat_num = tf.feature_column.numeric_column(key=feat, default_value=-1, dtype=tf.int32, normalizer_fn=None)
        feat_dense = tf.keras.layers.DenseFeatures(feat_num, trainable=False, name=feat)
        seed_feat_input = tf.keras.layers.Input(shape=(1,), name='seed_' + feat)
        cand_feat_input = tf.keras.layers.Input(shape=(1,), name='cand_' + feat)
        seed_numerical_inputs['seed_' + feat] = seed_feat_input
        cand_numerical_inputs['cand_' + feat] = cand_feat_input
        seed_numerical_features['seed_' + feat] = feat_dense({feat: seed_feat_input})
        cand_numerical_features['cand_' + feat] = feat_dense({feat: cand_feat_input})

    # 历史观察
    paste_watches_input = tf.keras.layers.Input(shape=(20,), dtype=tf.string, name='paste_watches')
    user_paste_watches = user_id_dense({"user_id": paste_watches_input})

    # embedding编码
    user_id_hash_embeddings = HashEmbedding(100000, 256, name='user_id_hash_embeddings')

    seed_user_id_embeddings = user_id_hash_embeddings(seed_user_id)
    cand_user_id_embeddings = user_id_hash_embeddings(cand_user_id)
    user_paste_watches_embeddings = user_id_hash_embeddings(user_paste_watches, mean=True)

    seed_features = tf.keras.layers.Concatenate(axis=-1, name='seed_features_concat')([
        seed_user_id_embeddings,
    ] + list(seed_numerical_features.values()))

    query_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='seed_concat_hist')([
        seed_features, user_paste_watches_embeddings
    ])

    query_x = tf.keras.layers.Dense(256, name='query_dense_0', kernel_initializer='he_uniform', )(query_tower_inputs)
    query_x = tf.keras.layers.PReLU(name='query_prelu_0')(query_x)
    query_x = tf.keras.layers.Dropout(0.4, name='query_dropout_0')(query_x)
    query_x = tf.keras.layers.Dense(128, name='query_dense_1', kernel_initializer='he_uniform')(query_x)
    query_x = tf.keras.layers.PReLU(name='query_prelu_1')(query_x)
    query_x = tf.keras.layers.Dropout(0.2, name='query_dropout_1')(query_x)
    query_x = L2Normalization(128, name='query_l2_norm')(query_x)

    candidate_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='cand_feature_cancat')([
        cand_user_id_embeddings,
    ] + list(cand_numerical_features.values()))
    candidate_x = tf.keras.layers.Dense(256, name='candidate_dense_0', kernel_initializer='he_uniform')(
        candidate_tower_inputs)
    candidate_x = tf.keras.layers.PReLU(name='candidate_prelu_0')(candidate_x)
    candidate_x = tf.keras.layers.Dropout(0.4, name='candidate_dropout_0')(candidate_x)
    candidate_x = tf.keras.layers.Dense(128, name='candidate_dense_1', kernel_initializer='he_uniform')(candidate_x)
    candidate_x = tf.keras.layers.PReLU(name='candidate_prelu_1')(candidate_x)
    candidate_x = tf.keras.layers.Dropout(0.2, name='candidate_dropout_1')(candidate_x)
    candidate_x = L2Normalization(128, name='candidate_l2_norm')(candidate_x)

    query_tower = tf.keras.Model(
        inputs=[
            seed_user_id_input,
        ] + list(seed_numerical_inputs.values()) + [paste_watches_input],
        outputs=query_x,
        name='query_tower'
    )

    candidate_tower = tf.keras.Model(
        inputs=[
            cand_user_id_input,
        ] + list(cand_numerical_inputs.values()),
        outputs=candidate_x,
        name='candidate_tower'
    )

    return query_tower, candidate_tower


    # # 历史输入(共享历史)
    # tf.feature_column.shared_embeddings([user_id_hash])

    # # 类别特征(非数值存储)转换onehot
    # thal_category = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
    # thal_onehot = tf.feature_column.indicator_column(thal_category)
    # features.append(thal_onehot)
    #
    # # 类别类别(非数值存储)转化为embedding
    # thal_embedding = tf.feature_column.embedding_column(thal_onehot, dimension=8)
    # features.append(thal_embedding)

    # # 多特征交叉，模型能够单独学习组合特征
    # cross_feature = tf.feature_column.crossed_column([age_bucket, thal_category], hash_bucket_size=1000)
    # cross_feature = tf.feature_column.indicator_column(cross_feature)
    # features.append(cross_feature)

    # return features

if __name__ == "__main__":
    query_tower, candidate_tower = build_model()
    query_tower.summary()
    candidate_tower.summary()
    tf.keras.utils.plot_model(query_tower, to_file="query_tower.png", show_shapes=True)
    tf.keras.utils.plot_model(candidate_tower, to_file="candidate_tower.png", show_shapes=True)