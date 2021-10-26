import sys, os
import pandas as pd
import datetime
from tensorflow.keras.callbacks import TensorBoard
path = os.path.dirname(__file__)

sys.path.append(path)

from .model import *


# 特征解析
def _parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim="\t")
    parsed = dict(zip(COL_NAME, item_feats))

    feature_dict = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                kvpairs = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                kvpairs = tf.strings.split(kvpairs, ":")
                kvpairs = kvpairs.to_tensor()
                feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                feat_vals = tf.reshape(feat_vals, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.float32)
                feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
                feature_dict[feat_col.name] = feat_ids
                feature_dict[feat_col.weight_name] = feat_vals
            else:
                feat_ids = tf.strings.split([parsed[feat_col.name]], ",").values[:feat_col.maxlen]
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                if feat_ids.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                feature_dict[feat_col.name] = feat_ids
        elif isinstance(feat_col, SpareFeat):
            feature_dict[feat_col.name] = parsed[feat_col.name]
        elif isinstance(feat_col, DenseFeat):
            if not feat_col.pre_embed:
                feature_dict[feat_col.name] = parsed[feat_col.name]
            elif feat_col.reduce_type is not None:
                keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                feature_dict[feat_col.name] = emb
            else:
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                feature_dict[feat_col.name] = emb
        else:
            raise Exception("unknown feature_columns....")

    # 分桶离散化
    for ft in BUCKET_DICT:
        feature_dict[ft] = tf.raw_ops.Bucketize(input=feature_dict[ft], boundaries=BUCKET_DICT[ft])

    label = parsed['label']
    duration = parsed['dur']

    return feature_dict, (label, duration)


if __name__ == "__main__":
    # 筛选有效的
    valid_keyword = pd.read_csv('/opt/data1/xululu/keyword_freq.csv', sep='\t')
    valid_keyword = valid_keyword[valid_keyword.cnt >= 2]
    
    # 筛选实体标签
    CATEGORICAL_MAP = {
        "keyword": valid_keyword.keyword_tag.unique().tolist(),
        "dislike_keyword": valid_keyword.keyword_tag.unique().tolist(),
        "most_topic": list(range(0, 710)),
        "dislike_topic": list(range(0, 710))
    }

    feature_columns = [
        DenseFeat(name="c_topic_id_ctr", pre_embed=None, reduce_type=None, dim=1, dtype="float32"),
        SpareFeat(name="user_id", voc_size=2, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype="int32"),
        SpareFeat(name="c_follow_topic_id", voc_size=2, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name="c_search_keyword", voc_size=2, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype="int32"),
        SpareFeat(name="exposure_hourdiff", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name="reply", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name="share", voc_size=6, hash_size= None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name="recommend", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name='topic_id', voc_size=720, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        SpareFeat(name='exposure_hour', voc_size=25, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
        VarLenSparseFeat(name="follow_topic_id", voc_size=720, hash_size=None, vocab=None, share_embed="topic_id", weight_name=None, combiner='sum', embed_dim=8, maxlen=20, dtype='int32'),
        VarLenSparseFeat(name="major_topic", voc_size=720, hash_size=None, vocab=None, share_embed='topic_id', weight_name=None, combiner='sum', embed_dim=8, maxlen=10, dtype='int32'),
        VarLenSparseFeat(name="keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed=None, weight_name=None, combiner='sum', embed_dim=8, maxlen=5, dtype='int32'),
        VarLenSparseFeat(name="search_keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed='keyword', weight_name=None, combiner='sum', embed_dim=12, maxlen=5, dtype='int32'),
        VarLenSparseFeat(name="major_keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed='keyword', weight_name=None, combiner='sum', embed_dim=8, maxlen=30, dtype='int32'),
        VarLenSparseFeat(name="topic_dislike_7d", voc_size=720, hash_size=None, vocab='dislike_topic', share_embed='dislike_topic', weight_name=None, combiner='sum', embed_dim=8, maxlen=7, dtype='int32'),
    ]

    # 用户特征及帖子特征
    dnn_feature_columns_name = [
        'c_follow_topic_id', 'c_search_keyword', 'c_topic_id_ctr',
        'c_major_topic_id', 'c_major_keyword', 'c_topic_dislike_7d',
        "topic_id", 'exposure_hour', "exposure_hourdiff", 'reply', 'recommend', 'keyword', "entity",
        'follow_topic_id', "search_keyword", 'major_keyword', 'major_topic', 'topic_dislike_7d',
    ]

    dnn_feature_columns = [col for col in feature_columns if col.name in dnn_feature_columns_name]

    # 离散分桶边界定义
    BUCKET_DICT = {
        'exposure_hourdiff': [3, 7, 15, 33],
        'reply': [12, 30, 63, 136],
        # 'share': [2, 11],
        'recommend': [1, 6, 16, 45],
    }

    DEFAULT_VALUES = [
        ['0'], [0], [0], [0.0], [0],
        [0], [0.0], [0], [0], [0],
        [0], [0], [0], [0], [0],
        ['0'], ['0'], ['0'], ['0'], ['0'],
        ['0'], ['0'],
    ]

    COL_NAME = [
        'user_id', 'post_id', 'label', 'dur', 'c_follow_topic_id',
        'c_search_keyword', 'c_topic_id_ctr', 'c_major_topic_id', 'c_major_keyword', 'c_topic_dislike_7d',
        'topic_id', 'exposure_hour', 'exposure_hourdiff', 'reply', 'recommend',
        'keyword', 'entity', 'follow_topic_id', 'search_keyword', 'major_topic',
        'major_keyword', 'topic_dislike_7d'
    ]

    pad_shapes = {}
    pad_values = {}

    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenSparseFeat):
            max_tokens = feat_col.maxlen
            pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.name] = "-1" if feat_col.dtype == 'string' else -1

            if feat_col.weight_name is not None:
                pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
                pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)
        # no need to pad labels
        elif isinstance(feat_col, SpareFeat):
            pad_values[feat_col.name] = '-1' if feat_col.dtype == 'string' else -1
            pad_shapes[feat_col.name] = tf.TensorShape([])
        elif isinstance(feat_col, DenseFeat):
            if not feat_col.pre_embed:
                pad_shapes[feat_col.name] = tf.TensorShape([])
            else:
                pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
            pad_values[feat_col.name] = 0.0

    pad_shapes = (pad_shapes, (tf.TensorShape([]), tf.TensorShape([])))
    pad_values = (pad_values, (tf.constant(0, dtype=tf.int32), tf.constant(0.0, dtype=tf.float32)))

    # 训练数据
    filenames = tf.data.Dataset.list_files(['./test_data.csv'])
    dataset = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    batch_size = 100
    daatset = dataset.map(_parse_function, num_parallel_calls=50)
    dataset = dataset.repeat()
    # 缓冲区随机打乱数据
    dataset = dataset.shuffle(buffer_size=batch_size)
    # 每1024条数据为一个batch，生成一个新的Datasets
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=pad_shapes, padding_values=pad_values)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ########################################################################
                   #################模型训练##############
    ########################################################################
    model = MMOE(dnn_feature_columns=dnn_feature_columns, num_tasks=2, tasks=['binary', 'regression'], tasks_name=['CTR', 'DUR'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.7, beta_2=0.8),
        loss={
            "CTR": "binary_crossentropy",
            "DUR": "mse"
        },
        loss_weights=[1.0, 1.0],
        metrics={
            "CTR": [tf.keras.metrics.AUC(name='auc')],
            "DUR": ["mae"]
        }
    )

    log_dir = './mywork/tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             write_graph=True,  # 是否存储网络结构图
                             write_images=True,  # 是否可视化参数
                             update_freq='epoch',
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None,
                             profile_batch='2,2')

    total_train_sample = 500
    total_test_sample = 50
    train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
    test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
    history_loss = model.fit(
        dataset, epochs=1,
        steps_per_epoch=train_steps_per_epoch,
        verbose=1, callbacks=[tbCallBack]
    )

