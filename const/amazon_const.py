import tensorflow as tf
import pickle

# TFRecord数据集储存格式
AMAZON_PROTO = {
    'reviewer_id': tf.io.FixedLenFeature([], tf.int64),
    'hist_item_list': tf.io.VarLenFeature(tf.int64),
    'hist_category_list': tf.io.VarLenFeature(tf.int64),
    'hist_length': tf.io.FixedLenFeature([], tf.int64),
    'item': tf.io.FixedLenFeature([], tf.int64),
    'item_category': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature([], tf.int64)
}

AMAZON_TARGET = 'target'

AMAZON_VARLEN = ['hist_item_list', 'hist_category_list']

with open('data/amazon/remap.pkl', 'rb') as f:
    _ = pickle.load(f)
    AMAZON_CATE_LIST = pickle.load(f)
    AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT, _ = pickle.load(f)

AMAZON_EMB_DIM = 64