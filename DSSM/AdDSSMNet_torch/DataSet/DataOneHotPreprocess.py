import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.style.use('ggplot')


class DSSMSomeFeaturesDataSet(nn.Module):

    def __init__(self, dataPath):
        super(DSSMSomeFeaturesDataSet, self).__init__()
        self.data_path = dataPath
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label = \
            self.samples[idx]
        return self.uid_codec.transform([uid]), self.age_codec.transform([age]), self.work_codec.transform([work_id]), self.height_codec.transform(
            [height]), self.sexs_codec.transform([sex]), self.uid_codec.transform([match_uid]), self.age_codec.transform([match_age]), self.work_codec.transform(
            [match_work_id]), self.height_codec.transform([match_height]), self.sexs_codec.transform([match_sex]), label

    def _init_dataset(self):
        df = pd.read_csv(self.data_path,
                         header=1,
                         names=["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age",
                                "match_user_work_id", "match_user_height", "match_user_sex", "label"],
                         low_memory=False)

        # 清洗数据
        df = self._clean_data(df, "label", ["uid", "subeventid"])

        # 获取数据标签
        self.label = torch.FloatTensor(df['label'].to_numpy()).view(-1, 1)

        # 筛选出不少于3条浏览记录的用户
        df = df[df.num >= 5]
        # print(df[df['label'] == 1].count())
        # print(df[df['label'] == 0].count())
        pos_data = df[df['label'] == 1][:20000]
        neg_data = df[df['label'] == 0][:30000]

        # print(pos_data[pos_data['label'] == 1].count())
        # print(neg_data[neg_data['label'] == 0].count())
        # print("--------------------------")
        df = pd.concat([pos_data, neg_data])

        # 筛选出用户和被匹配推荐人选择若干特征
        df = df[["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age", "match_user_work_id",
                 "match_user_height", "match_user_sex", "label"]]
        import random
        data = df.values
        random.shuffle(data)
        # 训练数据集
        self.samples = data.tolist()

        uids = set(df['uid'].tolist() + df['subeventid'].tolist())
        ages = set(df['age'].tolist() + df['match_user_age'].tolist())
        work_ids = set(df['work_id'].tolist() + df['match_user_work_id'].tolist())
        heights = set(df['height'].tolist() + df['match_user_height'].tolist())
        sexs = set(df['sex'].tolist() + df['match_user_sex'].tolist())

        self.uid_codec = self.convert_feature_to_scale()
        self.uid_codec.fit(list(uids))
        self.age_codec = self.convert_feature_to_scale()
        self.age_codec.fit(list(ages))
        self.work_codec = self.convert_feature_to_scale()
        self.work_codec.fit(list(work_ids))
        self.height_codec = self.convert_feature_to_scale()
        self.height_codec.fit(list(heights))
        self.sexs_codec = self.convert_feature_to_scale()
        self.sexs_codec.fit(list(sexs))

        # nunique_vals = list()
        # for column in df:
        #     nunique_vals.append(df[column].nunique())
        # pd.DataFrame({'columns': all_df.columns, 'num_of_unique': nunique_vals})

        self.user_field_dims = [len(uids), len(ages), len(work_ids), len(heights), len(sexs)]
        self.item_field_dims = [len(uids), len(ages), len(work_ids), len(heights), len(sexs)]

    def _clean_data(self, df, sorted_field, features):
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', None)

        # duplicationDF = df.groupby(["uid", "subeventid"]).agg({'label': 'count'}).reset_index()
        duplicationDF = df.groupby(["uid"]).agg({'label': 'count'})
        duplicationDF = duplicationDF.rename(columns={"label": "num"}).reset_index()

        df = df.sort_values(by=sorted_field)
        df = df.drop_duplicates(subset=features, keep='first')

        df = pd.merge(df, duplicationDF, how="left")

        # print(pd.merge(df, duplicationDF).head())
        # 统计次数
        # print(df.groupby(features).value_counts().unstack().plot(kind='bar', figsize=(20, 4)))
        # plt.show()

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

    def convert_feature_to_scale(self):
        feature_enc = LabelEncoder()
        return feature_enc

    def encode_continus_feature(self, x):
        feature_enc = StandardScaler()
        scaler = feature_enc.fit(x)
        return scaler

    def encode_discrete_feature(self):
        pass

    def encode_cyclical_feature(self, df):
        df['month_sin'] = np.sin((df['month'] - 1) * (2.0 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2.0 * np.pi / 12))

        df['day_sin'] = np.sin((df['day'] - 1) * (2.0 * np.pi / 7))
        df['day_cos'] = np.cos((df['day'] - 1) * (2.0 * np.pi / 7))


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    import string
    from torch.utils.data import DataLoader

    dataPath = "C:/Users/EDZ/Desktop/result.csv"
    dataset = DSSMSomeFeaturesDataSet(dataPath)

    # print(dataset.__len__())
    # setup_seed(20)
    dataLoader = DataLoader(dataset, batch_size=6, shuffle=True)
    # labelLoader = DataLoader(dataset.label, batch_size=1000, shuffle=True)
    # picEmbeddingLoader = DataLoader(dataset.pic_embedding, batch_size=1000, shuffle=True)
    #
    for batch_i, samples in enumerate(dataLoader):
    #     print("这个第几个batch_%s-------"%(str(batch_i)))
    #     uidAndMatchUid, pic_embedding, label = samples
        uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label = samples
        print(uid)
    #
    #     input1 = torch.cat((uid, age, work_id, height, sex, match_uid), dim=2).squeeze(1)
    #     print(input1.size())
    #     print(pic_embedding.size())
    #     print(label.size())

    # print(label)
    # print(next(iter(dataloader)))
