from collections import namedtuple, OrderedDict
import tensorflow as tf
import json

# 定义DSSM输入变量参数
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim',
                               'maxlen', 'dtype'])

feature_columns = [
    SparseFeat(name="topic_id", voc_size=700, hash_size=None, share_embed=None, embed_dim=16, dtype='string'),
    VarLenSparseFeat(name="most_post_topic_name", voc_size=700, hash_size=None, share_embed='topic_id',
                     weight_name=None, combiner='sum', embed_dim=16, maxlen=3, dtype='string'),
    VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, hash_size=None, share_embed='topic_id',
                     weight_name='all_topic_fav_7_weight', combiner='sum', embed_dim=16, maxlen=5, dtype='string'),
    DenseFeat(name='item_embed', pre_embed='post_id', reduce_type=None, dim=768, dtype='float32'),
    DenseFeat(name='client_embed', pre_embed='click_seq', reduce_type='mean', dim=768, dtype='float32'),
]

# 用户特征及贴子特征
user_feature_columns_name = ["all_topic_fav_7", "follow_topic_id", 'client_embed']
item_feature_columns_name = ['item_embed', "topic_id"]
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]


# 首先加载预训练 item embedding向量及离散特征vocabulary
def get_item_embed(file_names):
    item_bert_embed_dict = {}
    item_bert_embed = []
    item_bert_id = []
    for file in file_names:
        with open(file, 'r') as f:
            for line in f:
                feature_json = json.loads(line)
                tid = feature_json['tid']
                embedding = feature_json['features'][0]['layers'][0]['values']
                item_bert_embed_dict[tid] = embedding
    for k, v in item_bert_embed_dict.items():
        item_bert_id.append(k)
        item_bert_embed.append(v)

    item_id2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=item_bert_id,
            values=range(1, len(item_bert_id) + 1),
            key_dtype=tf.string,
            value_dtype=tf.int32),
        default_value=0)
    item_bert_embed = [[0.0] * 768] + item_bert_embed
    item_embedding = tf.constant(item_bert_embed, dtype=tf.float32)

    return item_id2idx, item_embedding


# user_id = get_client_id(ds)
ITEM_ID2IDX, ITEM_EMBEDDING = get_item_embed(file_names)

# 定义离散特征集合 ，离散特征vocabulary
DICT_CATEGORICAL = {
    "topic_id": [str(i) for i in range(0, 700)],
    "client_type": [0, 1]
}
