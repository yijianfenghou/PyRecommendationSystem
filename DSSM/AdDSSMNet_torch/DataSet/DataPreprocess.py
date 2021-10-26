from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class DSSMSomeFeaturesDataSet(nn.Module):

    def __init__(self, dataPath, charset, max_length):
        super(DSSMSomeFeaturesDataSet, self).__init__()
        self.data_path = dataPath
        self.charset = charset + '\0'
        self.max_length = max_length
        # self.samples = []
        self.uid_codec = LabelEncoder()
        self.age_codec = LabelEncoder()
        self.height_codec = LabelEncoder()
        self.work_id_codec = LabelEncoder()
        self.sex_codec = LabelEncoder()
        self.label_codec = LabelEncoder()
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, match_user_pic_embedding = \
        self.samples[idx]
        return self.label_encode(uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height,
                                 match_sex, label, match_user_pic_embedding)

    def _init_dataset(self):
        df = pd.read_excel(self.data_path,
                           names=["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age",
                                  "match_user_work_id", "match_user_height", "match_user_sex", "label", "pid",
                                  "match_user_pic_embedding"], sheet_name="Sheet2")

        # 清洗数据
        df = self._clean_data(df, "label", ["uid", "subeventid"])

        # 获取图片embedding那一列数据
        # pic_embedding_col = df['match_user_pic_embedding'].values
        # 变量转化成torch张量
        # self.pic_embedding = torch.from_numpy(
        #     np.array([[float(j) for j in i[1:-1].split(",")] for i in pic_embedding_col]))
        # df['match_user_pic_embedding'] = torch.from_numpy(np.array([[float(j) for j in i[1:-1].split(",")] for i in pic_embedding_col]))

        # 获取标签数据
        # labels = df['label'].to_numpy().reshape(-1, 1)
        # labels = set(labels)

        # # 获取数据标签
        # self.label = torch.FloatTensor(df['label'].to_numpy()).view(-1, 1)

        # 筛选出用户和被匹配推荐人选择若干特征
        df = df[["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age", "match_user_work_id",
                 "match_user_height", "match_user_sex", "label", "match_user_pic_embedding"]]

        # print(df.head())
        # 获取用户和被推荐人具有相同特征列
        self.samples = df.values.tolist()

        uids = set(df['uid'].tolist() + df['subeventid'].tolist())
        ages = set(df['age'].tolist() + df['match_user_age'].tolist())
        work_ids = set(df['work_id'].tolist() + df['match_user_work_id'].tolist())
        heights = set(df['height'].tolist() + df['match_user_height'].tolist())
        sexs = set(df['sex'].tolist() + df['match_user_sex'].tolist())
        labels = set(df['label'].tolist())

        self.uid_codec.fit(list(uids))
        self.age_codec.fit(list(ages))
        self.work_id_codec.fit(list(work_ids))
        self.height_codec.fit(list(heights))
        self.sex_codec.fit(list(sexs))

        self.label_codec.fit(list(labels))

    def _clean_data(self, df, sorted_field, features):
        df = df.sort_values(by=sorted_field)
        df = df.drop_duplicates(subset=features, keep='first')
        # 清除用户编号和改变用户类型
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
        # df['sex'] = df['sex'].astype("str")
        # df['match_user_sex'] = df['match_user_sex'].astype("str")
        df['sex'] = df['sex'].fillna('-1')
        df['match_user_sex'] = df['match_user_sex'].fillna('-1')

        return df

    # def split_train_validation(self, shuffle_dataset, validation_split):
    #     # if shuffle_dataset:
    #     #     np.random.seed(1)
    #     #     np.random.shuffle(self.samples)
    #     dataset_size = len(self.samples)
    #     train_size = int(0.8 * dataset_size)
    #     test_size = dataset_size - train_size
    #     self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.samples, [train_size, test_size])

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]

    def label_encode(self, uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height,
                     match_sex, label, match_user_pic_embedding):
        t_uid = torch.from_numpy(self.uid_codec.transform([uid]))
        t_age = torch.from_numpy(self.age_codec.transform([age]))
        t_work_id = torch.from_numpy(self.work_id_codec.transform([work_id]))
        t_height = torch.from_numpy(self.height_codec.transform([height]))
        t_sex = torch.from_numpy(self.sex_codec.transform([sex]))

        t_match_uid = torch.from_numpy(self.uid_codec.transform([match_uid]))
        t_match_age = torch.from_numpy(self.age_codec.transform([match_age]))
        t_match_work_id = torch.from_numpy(self.work_id_codec.transform([match_work_id]))
        t_match_height = torch.from_numpy(self.height_codec.transform([match_height]))
        t_match_sex = torch.from_numpy(self.sex_codec.transform([match_sex]))

        # t_label = self.to_one_hot(self.label_codec, [label])
        t_match_user_pic_embedding = torch.FloatTensor([float(j) for j in match_user_pic_embedding[1:-1].split(",")])

        return t_uid, t_age, t_work_id, t_height, t_sex, t_match_uid, t_match_age, t_match_work_id, t_match_height, t_match_sex, label, t_match_user_pic_embedding
        # return t_age, t_work_id, t_height, t_sex, t_match_age, t_match_work_id, t_match_height, t_match_sex, label, t_match_user_pic_embedding
        # return t_uid, t_age, t_match_uid, t_match_age, label, t_match_user_pic_embedding

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    import string
    from torch.utils.data import DataLoader

    dataPath = "C:/Users/EDZ/Desktop/result.xlsx"
    charset = string.ascii_letters + "-' "
    max_length = 30
    dataset = DSSMSomeFeaturesDataSet(dataPath, charset, max_length)

    # print(len(dataset))

    # dataloader = DataLoader(dataset=dataset, batch_size=6, shuffle=True)
    # print(dataset.__len__())
    # setup_seed(20)
    dataLoader = DataLoader(dataset, batch_size=6, shuffle=True)
    # labelLoader = DataLoader(dataset.label, batch_size=6, shuffle=True)
    # picEmbeddingLoader = DataLoader(dataset.pic_embedding, batch_size=6, shuffle=True)

    # for batch_i, samples in enumerate(zip(dataLoader, picEmbeddingLoader)):
    for batch_i, samples in enumerate(dataLoader):
        print("这个第几个batch_%s-------" % (str(batch_i)))
        uidAndMatchUid = samples
        uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, match_user_pic_embedding = uidAndMatchUid
        print(label)
    #
    #     label = label.squeeze(1)
    #     input1 = torch.cat((uid, age, work_id, height, sex), dim=1)
    #     print(input1)
    #     print("---------------------------")
    #     print(label)
    #     print(pic_embedding.size())
    #     print(label.size())

    # print(label)
    # print(next(iter(dataloader)))
