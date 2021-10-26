import os
import sys
from collections import OrderedDict
from tensorflow.keras.layers import Input, Concatenate, Flatten

path = os.path.dirname(__file__)
sys.path.append(path)

from .utils import *

# 定义model输入特征
def build_input_features(features_columns, prefix=''):

    input_features = OrderedDict()

    for feature_col in features_columns:

        if isinstance(feature_col, SpareFeat):
            input_features[feature_col.name] = Input([1], name=feature_col.name, dtype=feature_col.dtype)
        elif isinstance(feature_col, DenseFeat):
            input_features[feature_col.name] = Input([feature_col.dim], name=feature_col.name, dtype=feature_col.dtype)
        elif isinstance(feature_col, VarLenSparseFeat):
            input_features[feature_col.name] = Input([None], name=feature_col.name, dtype=feature_col.dtype)
            if feature_col.weight_name is not None:
                input_features[feature_col.weight_name] = Input([None], name=feature_col.weight_name, dtype=tf.float64)
        else:
            raise Exception("Invalid feature column in build_input_features: {}".format(feature_col.name))

    return input_features


# 构建自定义的embedding_matrix
def build_embedding_matrix(features_columns, linear_dim=None):
    embedding_matrix = {}
    for feature_col in features_columns:
        if isinstance(feature_col, SpareFeat) or isinstance(feature_col, VarLenSparseFeat):
            vocab_name = feature_col.share_embed if feature_col.share_embed else feature_col.name
            vocab_size = feature_col.voc_size + 2
            embed_dim = feature_col.embed_dim if linear_dim is None else 1
            name_tag = '' if linear_dim is None else "_linear"

            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0, stddev=0.001, dtype=tf.float32), trainable=True, name=vocab_name+'_embed'+name_tag)

    return embedding_matrix


# 构建自定义的embedding层
def build_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns)

    for feature_col in features_columns:
        if isinstance(feature_col, SpareFeat):
            vocab_name = feature_col.share_embed if feature_col.share_embed is not None else feature_col.name
            embedding_dict[vocab_name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name], name='emb_lookup_'+feature_col.name)
        elif isinstance(feature_col, VarLenSparseFeat):
            vocab_name = feature_col.share_embed if feature_col.share_embed is not None else feature_col.name
            if feature_col.combiner is not None:
                if feature_col.weight_name is not None:
                    embedding_dict[vocab_name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], combiner=feature_col.combiner, has_weight=True, name='emb_lookup_sparse_' + feature_col.name)
                else:
                    embedding_dict[vocab_name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], combiner=feature_col.combiner, name='emb_lookup_sparse_' + feature_col.name)
            else:
                embedding_dict[vocab_name] = EmbeddingLookup(embedding_matrix[vocab_name], name='emb_lookup_' + feature_col.name)

    return embedding_dict


# 构造 自定义embedding层
def build_linear_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns, linear_dim=1)
    name_tag = '_linear'

    for feat_col in features_columns:
        if isinstance(feat_col, SpareFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name], name='emb_lookup_' + feat_col.name + name_tag)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], combiner=feat_col.combiner, has_weight=True, name='emb_lookup_sparse_' + feat_col.name + name_tag)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], combiner=feat_col.combiner, name='emb_lookup_sparse_' + feat_col.name + name_tag)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name], name='emb_lookup_' + feat_col.name + name_tag)

    return embedding_dict


# dense与embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict, cate_map=None):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SpareFeat):
            _input = features[feat_col.name]
            if feat_col.vocab is not None:
                vocab_name = feat_col.share_embed if feat_col.share_embed is not None else feat_col.name
                keys = cate_map[vocab_name]
                _input = VocabLayer(keys)(_input)
            elif feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)
            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            _input = features[feat_col.name]
            if feat_col.vocab is not None:
                mask_val = '-1' if feat_col.dtype == 'string' else -1
                vocab_name = feat_col.share_embed if feat_col.share_embed is not None else feat_col.name
                keys = cate_map[vocab_name]
                _input = VocabLayer(keys)(_input)
            elif feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)

            if feat_col.combiner is not None:
                input_sparse = DenseToSparseTensor(mask_value=-1)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)
        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise Exception("dnn_feature_columns can not be empty list")


def get_linear_logit(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add(name='linear_logit')([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten(name='linear_logit')(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1, name='linear_logit')(dense_linear)
        return dense_linear_layer
    else:
        raise Exception("linear_feature_columns can not be empty list")
