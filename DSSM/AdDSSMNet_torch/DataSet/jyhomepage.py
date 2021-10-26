import torch
import torch.nn as nn
import pandas as pd


class JYHomePageData(nn.Module):

    def __init__(self, dataset_path):
        super(JYHomePageData, self).__init__()
        # skiprows=2,
        data = pd.read_csv(dataset_path, names=["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age",
                                  "match_user_work_id", "match_user_height", "match_user_sex", "label", "pid",
                                  "match_user_pic_embedding"])
        self._init_dataset(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, match_user_pic_embedding = \
        self.samples[idx]
        return self.label_encode(uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height,
                                 match_sex, label, match_user_pic_embedding)

    def _init_dataset(self, df):

        # 清洗数据
        df = self._clean_data(df, "label", ["uid", "subeventid"])

        # 筛选出用户和被匹配推荐人选择若干特征
        df = df[["uid", "age", "work_id", "height", "sex", "subeventid", "match_user_age", "match_user_work_id",
                 "match_user_height", "match_user_sex", "label"]]

        # 获取用户和被推荐人具有相同特征列
        self.samples = df.values.tolist()

        uids = set(df['uid'].tolist() + df['subeventid'].tolist())
        ages = set(df['age'].tolist() + df['match_user_age'].tolist())
        work_ids = set(df['work_id'].tolist() + df['match_user_work_id'].tolist())
        heights = set(df['height'].tolist() + df['match_user_height'].tolist())
        sexs = set(df['sex'].tolist() + df['match_user_sex'].tolist())
        labels = set(df['label'].tolist())

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
        df['sex'] = df['sex'].fillna('-1')
        df['match_user_sex'] = df['match_user_sex'].fillna('-1')

        return df

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
