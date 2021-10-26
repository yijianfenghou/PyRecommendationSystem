from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.layers import Input
from .utils import *


# 制作输入特征
def build_input_feature(feature_columns, prefix=""):
    input_features = OrderedDict()
    for feat_col in input_features:
        if isinstance(feat_col, Densefeat):
            if feat_col.pre_embed is None:
                input_features[feat_col.name] = Input([1], name=feat_col.name)
            else:
                input_features[feat_col.name] = Input([None], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            else:
                input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype=tf.float32)
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构建自定义的embeddimg matrix
def build_embedding_matrix(feature_columns):
    embedding_matrix = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                vocab_size = feat_col.voc_size
                embed_size = feat_col.embed_dim
                if vocab_name not in embedding_matrix:
                    embedding_matrix[vocab_name] = tf.Variable(initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_size), mean=0.0, stddev=0.0, dtype=tf.float32))
    return embedding_matrix


# 构造自定义的embedding层
def build_embedding_dict(feature_columns, embedding_matrix):
    embedding_dict = {}
    # embedding_matrix = build_embedding_matrix(feature_columns)
    for feat_col in feature_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name], name="EncodeMuiltiEmb_"+feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.weight_name is not None:
                embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], has_weight=True, name="EncodeMultiEmb_"+feat_col.name)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], name="EncodeMultiEmb_" + feat_col.name)
    return embedding_dict


# dense为embedding特征输入DICT_CATEGORICAL
def input_from_feature_columns(features, feature_columns, embedding_dict, dict_categorical):
    sparse_embedding_list = []
    dense_embedding_list = []
    for feat_col in feature_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = dict_categorical[vocab_name]
                _input_sparse = VocabLayer(keys)(features[feat_col.name])
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                _embed = embedding_dict[feat_col.name](_input_sparse)
            else:
                _embed = embedding_dict[feat_col.name](_input_sparse)
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                _weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                _embed = embedding_dict[feat_col.name]([_input_sparse, _weight_sparse])
            else:
                _embed = embedding_dict[feat_col.name](_input_sparse)
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, Densefeat):
            dense_embedding_list.append(features[feat_col.name])
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))
    return sparse_embedding_list, dense_embedding_list



