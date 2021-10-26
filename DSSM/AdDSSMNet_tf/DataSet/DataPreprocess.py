from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def gen_data_set(data, negsample=0):
    # 按照时间排个序
    data.sort_values("timestamp", inplace=True)
    item_ids = data["subenventid_idx"].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby("uid_idx")):
        pos_list = hist["subeventid_idx"].tolist()
        # rating_list = hist['label'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                # test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))

    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    return train_set, test_set


def get_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # 填补缺失值
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding="post", truncating="post", value=-1)

    # 标签 subeventid： 最后一次的点击视频id号
    train_model_input = {"uid_idx": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad, "hist_len": train_hist_len}

    # 添加用户信息
    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


class DataSetPreprocess(object):

    def __init__(self, dataPath, seq_len, negsample):
        self.data_path = dataPath
        self.columns = ["uid", "age", "work_id", "height", "sex", "hist_click_id", "subeventid", "match_user_age",
                        "match_user_work_id", "match_user_height", "match_user_sex", "label"]
        self.seq_len = seq_len
        self.negsample = negsample
        self._init_dataset()

    def _init_dataset(self):
        df = pd.read_csv(self.data_path,
                         header=1,
                         names=self.columns,
                         low_memory=False)

        # 清洗数据
        df = self._clean_data(df, "label", ["uid", "subeventid"])

        # 筛选出不少于3条浏览记录的用户
        df = df[df.num >= 5]

        # 筛选正负样本
        pos_data = df[df['label'] == 1][:20000]
        neg_data = df[df['label'] == 0][:30000]
        # 全部数据
        df = pd.concat([pos_data, neg_data])

        # 打乱数据集
        df = shuffle(df)

        # # 需要使用的特征
        # features = ["uid", "age", "work_id", "height", "sex", "hist_click_id", "subeventid", "match_user_age",
        #             "match_user_work_id",
        #             "match_user_height", "match_user_sex"]
        #
        # # 筛选出用户和被匹配推荐人选择若干特征
        # df = df[["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age", "match_user_work_id",
        #          "match_user_height", "match_user_sex", "label"]]

        # 数据集
        self.X = df[["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age", "match_user_work_id",
                     "match_user_height", "match_user_sex"]]

        y = df["label"]

        uidKey2index = self.add_index_column("uid", "subeventid")
        ageKey2index = self.add_index_column("age", "match_user_age")
        locKey2index = self.add_index_column("work_id", "match_user_work_id")
        heightKey2index = self.add_index_column("height", "match_user_height")
        sexKey2index = self.add_index_column("sex", "match_user_sex")

        self.input = X
        self.label = np.array(y)

    def build_portrait(self):
        # 构建用户画像
        user_profile = self.X[["uid_idx", "age_idx", "work_id_idx", "height_idx", "sex_idx"]].drop_duplicates("uid_idx")
        user_profile.set_index("uid_idx", inplace=True)

        # 构建物品画像
        item_profile = self.X[
            ["subeventid_idx", "match_user_age_idx", "match_user_work_id_idx", "match_user_height_idx",
             "match_user_sex_idx"]].drop_duplicates('subeventid_idx')

        # 用户历史点击用户序列
        user_clickUid_list = self.X.groupby("uid_idx")["subeventid_idx"].apply(list)

    # 增加用户和点击用户的index列
    def add_index_column(self, column_name, match_column_name):
        values = list(set(self.X[column_name].values.tolist() + self.X[match_column_name].values.tolist()))
        key2index = {value: idx for idx, value in enumerate(values)}
        self.X[f"{column_name}_idx"] = self.X[column_name].map(key2index)
        self.X[f"{match_column_name}_idx"] = self.X[match_column_name].map(key2index)
        return key2index

    def _clean_data(self, df, sorted_field, features):
        duplicationDF = df.groupby(["uid"]).agg({'label': 'count'})
        duplicationDF = duplicationDF.rename(columns={"label": "num"}).reset_index()

        df = df.sort_values(by=sorted_field)
        df = df.drop_duplicates(subset=features, keep='first')

        df = pd.merge(df, duplicationDF, how="left")

        # # 清除用户编号和改变用户类型
        df = df.dropna(subset=['subeventid'])
        df['subeventid'] = df['subeventid'].astype('int64')

        # 填充（清除）用户年龄中的缺失和改变类型
        df['age'] = df['age'].fillna('0')
        df['match_user_age'] = df['match_user_age'].fillna('0')
        df['match_user_age'] = df['match_user_age'].astype('int64')

        # 填充（清除）用户工作地点中的缺失和改变类型
        df = df.dropna(subset=['match_user_work_id'])
        df['match_user_work_id'] = df['match_user_work_id'].astype('int64')

        # 填充（清除）用户身高中的缺失和数据类型修改
        df['height'] = df['height'].fillna(0)
        df['match_user_height'] = df['match_user_height'].fillna('0')
        df['match_user_height'] = df['match_user_height'].astype("int64")

        # 填充（清除）用户性别中的缺失和数据类型修改
        df['sex'] = df['sex'].fillna('-1')
        df['match_user_sex'] = df['match_user_sex'].fillna('-1')

        return df


def setup_seed(seed):
    import random
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    dataPath = "C:/Users/EDZ/Desktop/result.csv"

    dataset = DataSetPreprocess(dataPath)

    features = build_input_features(feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    query_feature_columns = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_feature_names:
            query_feature_columns.append(fc)

    key_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_names))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            key_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())
