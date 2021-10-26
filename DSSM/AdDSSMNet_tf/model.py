import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from DSSM.AdDSSMNet_tf.DataSet.DataPreprocess import *
from DSSM.AdDSSMNet_tf.DataSet.DataLoad import *
from impala.dbapi import connect
import datetime
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

def some_last_days(days):
    today = datetime.datetime.now()
    offset = datetime.timedelta(days=-days)
    beforeDate = (today + offset).strftime('%Y-%m-%d')
    return beforeDate

getActiveUsers = """
    select
        /*+ BROADCAST(dataplatform_user_action_record)*/
        t6.id,
        t1.age,
        t1.work_location,
        t1.height,
        t1.sex,
        nvl(t5.type, "2") type
    from
        (
            select
                uid,
                cast(year(now()) as int) - cast(birth_year as int) age,
                work_location,
                height,
                sex
            from
                algorithm.users_search
            where
                gid >= 3.0
                and gid <= 10.0
                and uid is not NULL
                and work_location = 51
            group by
                uid, cast(year(now()) as int) - cast(birth_year as int), work_location, height, sex
         )t1
    inner join
        (
             select
                 uid
             from
                algorithm.dataplatform_user_action_record
             where
                dt >= "%s"
                and eventid in ('8.32', '8.33', '8.34')
                and subeventid is not NULL
                and subeventid != ''
             group by
                uid
        )t2
        on cast(t1.uid as string) = t2.uid
    left join
        (
            select
                user_id,
                service_id,
                type
            from
                (
                    select
                        user_id,
                        service_id,
                        type,
                        row_number() over(partition by user_id order by type) num
                    from
                    (
                        select
                            user_id,
                            service_id,
                            case when service_id = '99' and times = 0 then '2' else type end type
                        from
                        (
                            select
                                user_id,
                                trim(service_id) service_id,
                                times
                            from
                               algorithm.jy_user_service
                        )t3
                        left join
                        (
                            select
                                id,
                                trim(type) type
                            from
                                algorithm.jy_user_service_type
                        )t4
                        on cast(t3.service_id as string) = t4.id
                )tmp1
           )tmp2
           where
               tmp2.num = 1
        )t5
    on cast(t1.uid as string) = cast(t5.user_id as string)
    inner join
        (
            select
                *
            from
                algorithm.uid_convert_to_id
        )t6
    on cast(t1.uid as string) = cast(t6.uid as string)
""".format(some_last_days(90))


# 自定损失函数，加权交叉熵损失
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, pos_weight=1.2, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.from_logits = from_logits


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.BinaryCrossentropy(y_true, y_pred, from_logits=self.from_logits)[:, None]
        ce = ce * (1 - y_true) + self.pos_weight * ce * (y_true)

        return ce

    def get_config(self, ):
        config = {'pos_weight': self.pos_weight, 'from_logits': self.from_logits, 'name': self.name}
        base_config = super(WeightedBinaryCrossEntropy, self).get_config()
        return base_config.uptate(config)


class Similarity(tf.keras.layers.Layer):
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
        return cosine_score
        # return tf.expand_dims(cosine_score, 1)
    def compute_output_shape(self, input_shape):
        return (None, 1)
    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


class L2Normalization(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self._weights = self.add_weight(
            shape=(self.kernel_size, ),
            initializer=initializers.glorot_normal,
            trainable=True,
            name="l2norm_weight"
        )
        self._bias = self.add_weight(
            shape=(self.kernel_size, ),
            initializer=initializers.zeros,
            trainable=True,
            name='l2norm_bias'
        )

    def call(self, inputs, **kwargs):
        outputs = self._weights*inputs + self._bias
        return K.l2_normalize(outputs)

def log_q(x, y, sampling_p=None, temperature=0.05):
    """logQ correction used in sampled softmax model."""
    inner_product = tf.reduce_sum(tf.math.multiply(x, y), axis=-1) / temperature
    if sampling_p is not None:
        return inner_product - tf.math.log(sampling_p)
    return inner_product

def sampling_p_estimation_single_hash(array_a, array_b, hash_indexs, global_step, alpha=0.01):
    """单Hash函数采样概率估计"""
    array_b[hash_indexs] = (1 - alpha) * array_b[hash_indexs] + \
                                alpha * (global_step - array_a[hash_indexs])
    array_a[hash_indexs] = global_step
    sampling_p = 1 / array_b[hash_indexs]
    return array_a, array_b, sampling_p

def corrected_batch_softmax(x, y, sampling_p=None):
    """logQ correction softmax"""
    corrected_inner_product = log_q(x, y, sampling_p=sampling_p)
    return tf.math.exp(corrected_inner_product)/tf.math.reduce_sum(tf.math.exp(corrected_inner_product))

def reward_corss_entropy(reward, output):
    # return -tf.reduce_mean(reward*tf.math.log(output))
    return -reward*tf.math.log(output)

def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_embedding_concat = concat_func(sparse_embedding_list)
        sparse_input_dim = sparse_embedding_concat.shape[2]
        dense_embedding_concat = concat_func(dense_value_list)
        dense_input_dim = dense_embedding_concat.shape[2]
        sparse_dnn_flatten = tf.keras.layers.Flatten()(sparse_embedding_concat)
        dense_dnn_flatten = tf.keras.layers.Flatten()(dense_embedding_concat)
        sparse_embedding_input = tf.reshape(sparse_dnn_flatten, [-1, sparse_input_dim])
        dense_embedding_input = tf.reshape(dense_dnn_flatten, [-1, dense_input_dim])
        return concat_func([sparse_embedding_input, dense_embedding_input])
    elif len(sparse_embedding_list) > 0:
        sparse_embedding_concat = concat_func(sparse_embedding_list)
        sparse_input_dim = sparse_embedding_concat.shape[2]
        sparse_embedding_flatten = tf.keras.layers.Flatten()(sparse_embedding_concat)
        sparse_embedding_reshape = tf.reshape(sparse_embedding_flatten, [-1, sparse_input_dim])
        return sparse_embedding_reshape
    elif len(dense_value_list) > 0:
        dense_embedding_concat = concat_func(dense_value_list)
        dense_input_dim = dense_embedding_concat.shape[2]
        dense_dnn_flatten = tf.keras.layers.Flatten()(dense_embedding_concat)
        dense_embedding_input = tf.reshape(dense_dnn_flatten, [-1, dense_input_dim])
        return dense_embedding_input
    else:
        raise Exception("dnn_feature_columns can not be empty list")


# 加工用户类型以及匹配条件
class UserAndMatchCond:

    def __init__(self, host="10.1.1.244", port=21050, database="algorithm"):
        self.host = host
        self.port = port
        self.database = database

    def get_user_type_sex(self):
        from impala.dbapi import connect
        conn = connect(host=self.host, port=self.port, database=self.database)
        # 定点游标
        cursor = conn.cursor()
        hive_sentence = getActiveUsers
        cursor.execute(hive_sentence)
        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        df = pd.DataFrame(result)
        return df


def DSSMModel(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(256, 256, 128), item_dnn_hidden_units=(256, 256,  128), out_dnn_activation='tanh', gamma=1.2, metric='cos'):
    '''
    函数式API搭建双塔模型
    '''
    feature_columns = user_feature_columns + item_feature_columns
    # 构建embedding_dict
    # embedding_metrix = build_embedding_matrix(feature_columns)
    embedding_dict = build_embedding_dict(feature_columns)

    # user特征处理
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns, embedding_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # item特征处理
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features, item_feature_columns, embedding_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    # user tower
    for i in range(len(user_dnn_hidden_units)):
        if i == len(user_dnn_hidden_units)-1:
            user_dnn_out = tf.keras.layers.Dense(units=user_dnn_hidden_units[i], activation=out_dnn_activation, name="user_embed_out")(user_dnn_input)
        user_dnn_input = tf.keras.layers.Dense(units=user_dnn_hidden_units[i], activation='relu', name='dnn_user_'+str(i))(user_dnn_input)

    # item tower
    for i in range(len(item_dnn_hidden_units)):
        if i == len(item_dnn_hidden_units)-1:
            item_dnn_out = tf.keras.layers.Dense(units=item_dnn_hidden_units[i], activation=out_dnn_activation, name='item_embed_out')(item_dnn_input)
        item_dnn_input = tf.keras.layers.Dense(units=item_dnn_hidden_units[i], activation='relu', name='dnn_item_'+str(i))(item_dnn_input)

    score = Similarity(type_sim=metric, gamma=gamma)([user_dnn_out, item_dnn_out])
    output = tf.keras.layers.Activation("sigmoid", name="dssm_out")(score)

    # uid = tf.keras.layers.Input(shape=(None,), name="uid")
    # age = tf.keras.layers.Input(shape=(None,), name="age")
    # work_id = tf.keras.layers.Input(shape=(None,), name="work_id")
    # height = tf.keras.layers.Input(shape=(None,), name="height")
    # sex = tf.keras.layers.Input(shape=(None,), name="sex")
    # hist_tuid_record = tf.keras.layers.Input()
    # subeventid = tf.keras.layers.Input(shape=(None,), name="subeventid")
    # match_user_age = tf.keras.layers.Input(shape=(None,), name="match_user_age")
    # match_user_work_id = tf.keras.layers.Input(shape=(None,), name="match_user_work_id")
    # match_user_height = tf.keras.layers.Input(shape=(None,), name="match_user_height")
    # match_user_sex = tf.keras.layers.Input(shape=(None,), name="match_user_sex")
    # label = tf.keras.layers.Input(shape=(None,), name="label")
    #
    # # user塔
    # # user_vector = tf.keras.layers.concatenate([
    # #     layers.Embedding(num_uid, 100)(uid),
    # #     layers.Embedding(num_age, 8)(age),
    # #     layers.Embedding(num_work_id, 2)(work_id),
    # #     layers.Embedding(num_height, 8)(height),
    # #     layers.Embedding(num_sex, 4)(sex)
    # # ])
    # user_vector = tf.keras.layers.concatenate([
    #     layers.Embedding(num_uid, 100)(uid),
    #     layers.Embedding(num_work_id, 2)(work_id)
    # ])
    # user_vector = layers.Dense(128, activation='tanh')(user_vector)
    # user_vector = layers.Dense(8, activation='tanh', kernel_regularizer='l2')(user_vector)
    #
    # # movie塔
    # # movie_vector = tf.keras.layers.concatenate([
    # #     layers.Embedding(num_uid, 100)(subeventid),
    # #     layers.Embedding(num_age, 8)(match_user_age),
    # #     layers.Embedding(num_work_id, 2)(match_user_work_id),
    # #     layers.Embedding(num_height, 8)(match_user_height),
    # #     layers.Embedding(num_sex, 4)(match_user_sex)
    # # ])
    # movie_vector = tf.keras.layers.concatenate([
    #     layers.Embedding(num_uid, 100)(subeventid),
    #     layers.Embedding(num_work_id, 2)(match_user_work_id)
    # ])
    # movie_vector = layers.Dense(128, activation='tanh')(movie_vector)
    # movie_vector = layers.Dense(8, activation='tanh', kernel_regularizer='l2')(movie_vector)
    #
    # # 计算两个塔出来向量的相似度softmax值
    # user_movie_softmax = corrected_batch_softmax(user_vector, movie_vector, sampling_p=0.05)
    #
    # # 计算两个塔相似度与标签的关系
    # loss = reward_corss_entropy(label, user_movie_softmax)
    # loss = tf.reduce_mean(loss, name="loss")
    # output = tf.keras.layers.Activation("sigmoid", name="dssm_out")(score)

    # 每个用户的embedding和item的embedding作点积
    # score = Similarity(type_sim=metric, gamma=gamma)([user_vector, movie_vector])

    # dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1)
    # dot_user_movie = tf.expand_dims(dot_user_movie, 1)
    #
    # output = layers.Dense(1, activation='sigmoid')(dot_user_movie)
    # output = tf.keras.layers.Activation("softmax", name="dssm_out")(score)

    model = tf.keras.models.Model(inputs=user_inputs_list + item_inputs_list, outputs=[output])
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


def visual_loss_valid_loss(model):
    import matplotlib.pyplot as plt

    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def get_user_id_hive(conn):
    # 定点游标
    cursor = conn.cursor()
    hive_sentence = "select * from algorithm.uid_convert_to_id"
    cursor.execute(hive_sentence)
    columns = [col[0] for col in cursor.description]
    result = [dict(zip(columns, row)) for row in cursor.fetchall()]
    df = pd.DataFrame(result)
    df.columns = ["uid", "uidLabelEncoder"]
    return df


def get_sichuan_user_hive(conn):
    # 定点游标
    cursor = conn.cursor()
    hive_sentence = getActiveUsers
    cursor.execute(hive_sentence)
    columns = [col[0] for col in cursor.description]
    result = [dict(zip(columns, row)) for row in cursor.fetchall()]
    df = pd.DataFrame(result)
    return df


if __name__ == "__main__":

    DEFAULT_VALUES = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [0], ['']]
    COL_NAME = ['uid', 'age', 'workid', 'height', 'sex', 'timeStamp', 'tuid', 'tage', "tworkid", 'theight', 'tsex', 'label', 'watched_history_uid_list']
    # dataPath = "/var/chenhaolin/remotedata/dataset/*"
    dataPath = "C:/Users/EDZ/Desktop/dataset/*"
    dataLoad = DataLoad(dataPath, DEFAULT_VALUES, COL_NAME, feature_columns)
    dataset = dataLoad.dataset

    model = DSSMModel(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(256, 256, 128), item_dnn_hidden_units=(256, 256, 128), gamma=1)

    # model.compile(loss={"dssm_out": WeightedBinaryCrossEntropy()}, loss_weights=[1.0], optimizer=tf.keras.optimizers.RMSprop(), metrics={"dssm_out": tf.keras.metrics.AUC(name="auc")})
    model.compile(loss={"dssm_out": WeightedBinaryCrossEntropy()}, loss_weights=[1.0], optimizer=tf.keras.optimizers.RMSprop(), metrics={"dssm_out": tf.keras.metrics.AUC(name="auc")})

    # patience值用来检查改进epochs 的数量
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=0)

    # 训练数据集和验证集大小
    batch_size = 1024
    total_train_sample = sum(1 for _ in dataset)

    # train_dataset = dataset.enumerate().filter(lambda x, y: x % 4 != 0).map(lambda x, y: y)
    # test_dataset = dataset.enumerate().filter(lambda x, y: x % 4 == 0).map(lambda x, y: y)

    total_test_sample = 1212
    train_steps_per_epoch = np.floor(total_train_sample/batch_size).astype(np.int32)
    test_steps_per_epoch = np.floor(total_test_sample/batch_size).astype(np.int32)

    # 显示进度条
    from datetime import datetime
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/logs_" + TIMESTAMP)

    history = model.fit(dataset, epochs=1, steps_per_epoch=train_steps_per_epoch, verbose=1, validation_split=0.2, callbacks=[early_stop, tensorboard_callback])

    user_embedding_model = tf.keras.models.Model(
        inputs=model.user_input,
        outputs=model.user_embedding
    )
    # 保存用户侧的模型
    user_embedding_model.save("./output/models/dssmUser/user_embedding_model.h5")

    # 得到item embeddings的值
    item_embedding_model = tf.keras.models.Model(
        inputs=model.item_input,
        outputs=model.item_embedding
    )
    # 保存推荐侧的模型
    # tf.keras.models.save_model(item_embedding_model, "./output/models/dssmItem/")
    item_embedding_model.save("./output/models/dssmItem/item_embedding_model.h5")

    # # 加载模型文件
    # user_embedding_model = tf.keras.models.load_model("./output/models/dssmUser/user_embedding_model.h5")

    # # 保存用户的embedding值,遍历Dataframe版本
    # user_embeddings = []
    # for index, row in df_user.iterrows():
    #     user_id = row["UserID"]
    #     user_input = [
    #         np.reshape(row["UserID_idx"], [1, 1]),
    #         np.reshape(row["Gender_idx"], [1, 1]),
    #         np.reshape(row["Age_idx"], [1, 1]),
    #         np.reshape(row["Occupation_idx"], [1, 1])
    #     ]
    #     user_embedding = user_layer_model(user_input)
    #
    #     embedding_str = ",".join([str(x) for x in user_embedding.numpy().flatten()])
    #     user_embeddings.append([user_id, embedding_str])
    # df_user_embedding = pd.DataFrame(user_embeddings, columns=["user_id", "user_embedding"])
    # df_user_embedding.head()
    # output = "./datas/ml-latest-small/tensorflow_user_embedding.csv"
    # df_user_embedding.to_csv(output, index=False)

    conn = connect(host="10.1.1.244", port=21050, database="algorithm")

    columns = ["uidLabelEncoder", "uid_embedding"]
    uid_embedding_list = []
    uid_embedding_dict = {}
    for i, data in enumerate(dataset):
        input = data[0]
        uid = input["uid"].numpy()[0]
        if uid not in uid_embedding_dict:
            uid_embedding_dict[uid] = 1
            user_query = {"uid": input["uid"], "age": input["age"], "workid": input["workid"], "height": input["height"], "sex": input["sex"], "watched_history_uid_list": input["watched_history_uid_list"]}
            user_embs = user_embedding_model.predict(user_query)[0]
            uid_embedding_list.append(dict(zip(columns, [uid, user_embs])))

    uid_result = pd.DataFrame(uid_embedding_list)
    uid_result.columns = columns
    uid_result['uidLabelEncoder'] = uid_result['uidLabelEncoder'].astype("int64")

    uid_hive_labelEncoder = get_user_id_hive(conn)
    uid_hive_labelEncoder['uidLabelEncoder'] = uid_hive_labelEncoder['uidLabelEncoder'].astype("int64")

    uid_result = uid_result.merge(uid_hive_labelEncoder, on="uidLabelEncoder", how="inner")[["uid", "uid_embedding"]]

    # user_id = np.array(XDF["uid"])
    # user_input = [
    #     np.reshape(np.array(XDF["uid_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["age_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF['work_id_idx']), [-1, 1]),
    #     np.reshape(np.array(XDF["height_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["sex_idx"]), [-1, 1]),
    # ]
    # user_embedding = user_embedding_model(user_input)
    #
    # user_embedding = tf.reshape(user_embedding, [-1, 8])
    #
    # embedding_str = [",".join([str(x) for x in each_uid_embedding]) for each_uid_embedding in user_embedding.numpy()]
    #
    # # 保存的列名
    # columnNames = ["user_id", "user_embedding"]
    # # 保存用户的embeddings值
    # user_embedding = {
    #     "user_id": user_id,
    #     "user_embedding": embedding_str
    # }
    #
    # df_user_embedding = pd.DataFrame(user_embedding)

    user_output = "./output/result/tensorflow_user_embedding.csv"
    uid_result.to_csv(user_output, index=False)

    item_id = get_sichuan_user_hive(conn)

    age_dict = {i: str(i + 1) for i in range(100)}
    workid_dict = {i: str(i + 1) for i in range(100)}
    sex_dict = {'f': '1', 'm': '2', '-1': '0'}
    height_dict = {i: str(i - 99) for i in range(100, 227)}

    # columns = ["uidLabelEncoder", "item_embedding"]
    # item_embedding_list = []

    # 过滤掉不符合条件的数值
    item_id = item_id.loc[(item_id.age >= 0) & (item_id.age <= 99)]
    item_id = item_id.loc[(item_id.work_location >= 0) & (item_id.work_location <= 99)]
    item_id = item_id.loc[(item_id.height >= 100) & (item_id.height <= 226)]

    item_id["age"] = item_id["age"].map(lambda x: age_dict[x])
    item_id["work_location"] = item_id["work_location"].map(lambda x: workid_dict[x])
    item_id["sex"] = item_id["sex"].map(lambda x: sex_dict[x])
    item_id["height"] = item_id["height"].map(lambda x: height_dict[x])

    tuids_list = item_id['id'].values.reshape(-1, 1)
    tuids = tf.convert_to_tensor(tuids_list, dtype=tf.string)
    tages = tf.convert_to_tensor(item_id['age'].values.reshape(-1, 1), dtype=tf.string)
    tworkids = tf.convert_to_tensor(item_id['work_location'].values.reshape(-1, 1), dtype=tf.string)
    theights = tf.convert_to_tensor(item_id['height'].values.reshape(-1, 1), dtype=tf.string)
    tsexs = tf.convert_to_tensor(item_id['sex'].values.reshape(-1, 1), dtype=tf.string)
    item_query = {"tuid": tuids, "tage": tages, "tworkid": tworkids,
                  "theight": theights, "tsex": tsexs}
    item_embs = item_embedding_model.predict(item_query)

    # for index, row in item_id.iterrows():
    #     # input = data[0]
    #     # tuid = input["tuid"].numpy()[0]
    #     # user_query = {"tuid": input["tuid"], "tage": input["tage"], "tworkid": input["tworkid"], "theight": input["theight"],
    #     #               "tsex": input["tsex"]}
    #     # item_embs = item_embedding_model.predict(user_query)[0]
    #     # item_embedding_list.append(dict(zip(columns, [tuid, item_embs])))
    #     tuid = row['id']
    #     try:
    #         tage = tf.constant([age_dict[row["age"]]], dtype=tf.string, shape=(1,))
    #         tworkid = tf.constant([workid_dict[row["work_location"]]], dtype=tf.string, shape=(1,))
    #         theight = tf.constant([height_dict[row["height"]]], dtype=tf.string, shape=(1,))
    #         tsex = tf.constant([sex_dict[row["sex"]]], dtype=tf.string, shape=(1,))
    #         item_query = {"tuid": tf.constant([tuid], dtype=tf.string, shape=(1,)), "tage": tage, "tworkid": tworkid, "theight": theight, "tsex": tsex}
    #         item_embs = item_embedding_model.predict(item_query)[0]
    #         item_embedding_list.append(dict(zip(columns, [tuid, item_embs])))
    #     except Exception as e:
    #         print("字段不符合逻辑，排除掉")
    item_result = pd.DataFrame({'uidLabelEncoder': tuids_list.reshape(-1), 'item_embedding': list(item_embs)}, columns=['uidLabelEncoder', 'item_embedding'])
    # item_result = pd.DataFrame(item_embedding_list)
    item_result['uidLabelEncoder'] = item_result['uidLabelEncoder'].astype("int64")

    item_result = item_result.merge(uid_hive_labelEncoder, on="uidLabelEncoder", how="inner")[["uid", "item_embedding"]]

    item_output = "./output/result/tensorflow_item_embedding.csv"
    item_result.to_csv(item_output, index=False)
    # 得到item embeddings的值
    # item_embedding_model = tf.keras.models.Model(
    #     inputs=model.item_input,
    #     outputs=model.item_embedding
    # )
    # 保存推荐侧的模型
    # tf.keras.models.save_model(item_embedding_model, "./output/models/dssmItem/")
    # item_embedding_model.save("./output/models/dssmItem/item_embedding_model.h5")
    #
    # movie_embeddings = []
    # for index, row in df_movie.iterrows():
    #     movie_id = row["MovieID"]
    #     movie_input = [
    #         np.reshape(row["MovieID_idx"], [1, 1]),
    #         np.reshape(row["Genres_idx"], [1, 1])
    #     ]
    #     movie_embedding = item_embedding_model(movie_input)
    #
    #     embedding_str = ",".join([str(x) for x in movie_embedding.numpy().flatten()])
    #     movie_embeddings.append([movie_id, embedding_str])
    #
    # df_movie_embedding = pd.DataFrame(movie_embeddings, columns=["movie_id", "movie_embedding"])
    #
    # output = "./datas/ml-latest-small/tensorflow_movie_embedding.csv"
    # df_movie_embedding.to_csv(output, index=False)

    # item_id = np.array(XDF["subeventid"])
    # item_input = [
    #     np.reshape(np.array(XDF["subeventid_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["match_user_age_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["match_user_work_id_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["match_user_height_idx"]), [-1, 1]),
    #     np.reshape(np.array(XDF["match_user_sex_idx"]), [-1, 1]),
    # ]
    # item_embedding = item_embedding_model(user_input)
    #
    # item_embedding = tf.reshape(item_embedding, [-1, 8])
    #
    # embedding_str = [",".join([str(x) for x in each_item_embedding]) for each_item_embedding in item_embedding.numpy()]
    #
    # # 保存用户的embeddings值
    # item_embedding = {
    #     "item_id": item_id,
    #     "item_embedding": embedding_str
    # }
    #
    # df_user_embedding = pd.DataFrame(item_embedding)
    #
    # output = "./output/result/tensorflow_item_embedding.csv"
    # df_user_embedding.to_csv(output, index=False)