import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def gen_data_set(userData, itemData, sexCandidateDict, negsample=0):
    # 某区域的全部用户
    allUser = itemData["tuid"].values.tolist()

    train_set = []
    test_set = []
    for uid, oneUserData in tqdm(userData.groupby('uid')):
        # 曝光用户特征
        user_features = oneUserData.values[0][1:-3]
        # 曝光样本的集合
        user_item_list = oneUserData['subeventid'].tolist()
        # 曝光样本的正样本
        pos_list = oneUserData[oneUserData.label == 1]['subeventid'].tolist()
        # 曝光样本的负样本
        neg_list = oneUserData[oneUserData.label == 0]['subeventid'].tolist()

        # 额外的负样本采样，扩展到整个数据集序列
        if negsample > 0:
            userSex = user_features[-1]
            items_ids = sexCandidateDict[userSex]
            candidate_set = list(set(items_ids) - set(user_item_list))
            neg_list += list(np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True))

        # 每次采样负样本的数目
        epochNegSampleNum = len(neg_list) // (len(pos_list) + 1)
        for i in range(len(pos_list)):
            hist = pos_list[:i]
            candidateUser = pos_list[i]
            # print("--------------------")
            # print(candidateUser)
            # 得到候选用户特征
            if candidateUser in allUser:
                pos_candidate_user_feature = itemData[itemData.tuid == candidateUser].values.tolist()[0][1:]
                if i != len(pos_list)-1:
                    train_set.append((uid, user_features[0], user_features[1], user_features[2], user_features[3], hist, candidateUser, pos_candidate_user_feature[0], pos_candidate_user_feature[1], pos_candidate_user_feature[2], pos_candidate_user_feature[3], 1))
                    for negi in range(epochNegSampleNum):
                        if neg_list[epochNegSampleNum*i+negi] in allUser:
                            neg_candidate_user_feature = itemData[itemData.tuid == neg_list[epochNegSampleNum*i+negi]].values.tolist()[0][1:]
                            train_set.append((uid, user_features[0], user_features[1], user_features[2], user_features[3], hist, neg_list[epochNegSampleNum*i+negi], neg_candidate_user_feature[0], neg_candidate_user_feature[1], neg_candidate_user_feature[2], neg_candidate_user_feature[3], 1))
                else:
                    test_set.append((uid, user_features[0], user_features[1], user_features[2], user_features[3], hist, candidateUser, pos_candidate_user_feature[0], pos_candidate_user_feature[1], pos_candidate_user_feature[2], pos_candidate_user_feature[3], 1))

    return train_set, test_set


def train_input_fn(train_set, seq_max_len, batch_size):

    uid = np.array([line[0] for line in train_set])
    uage = np.array([line[1] for line in train_set])
    uworkid = np.array([line[2] for line in train_set])
    uheight = np.array([line[3] for line in train_set])
    usex = np.array([line[4] for line in train_set])
    watches_hist = np.array([line[5] for line in train_set])
    tid = np.array([line[6] for line in train_set])
    tage = np.array([line[7] for line in train_set])
    tworkid = np.array([line[8] for line in train_set])
    theight = np.array([line[9] for line in train_set])
    tsex = np.array([line[10] for line in train_set])
    labels = np.array([line[11] for line in train_set])

    # 填补缺失值
    # watches_hist = pad_sequences(watches_hist, maxlen=seq_max_len, padding="post", truncating="pre", value='0')
    # 模型输入
    user_model_input = {
        "seed_user_id": uid,
        "seed_age": uage,
        "seed_workid": uworkid,
        "seed_height": uheight,
        "seed_sex": usex,
        "paste_watches": watches_hist,
    }

    item_model_input = {
        "cand_user_id": tid,
        "cand_age": tage,
        "cand_workid": tworkid,
        "cand_height": theight,
        "cand_sex": tsex
    }

    dataset = tf.data.Dataset.from_tensor_slices((user_model_input, item_model_input, labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset


if __name__ == "__main__":
    # 读取用户侧数据
    userDF = pd.read_csv("C:/Users/EDZ/Desktop/user.csv")

    # 用户侧数据timeStamp
    # userDF['candidate_set'] = userDF['candidate_set'].str.split("&").apply(lambda x: list(set(x)))
    userDF['timeStamp'] = pd.to_datetime(userDF['timeStamp'])
    userDF = userDF.sort_values(by='timeStamp')

    # 构建用户画像
    user_profile = userDF[['uid', 'age', 'workid', 'height', 'sex']].drop_duplicates('uid')

    # 用户历史点击文章序列
    unDuplicationClickTuid = userDF[userDF.label == 1].drop_duplicates(["uid", "subeventid"])[:100]
    user_click_item_list = unDuplicationClickTuid.groupby("uid")["subeventid"].apply(list).to_frame().to_dict()
    user_click_item_dict = user_click_item_list['subeventid']

    # 用户曝光未点击的数据
    unDuplicationUnClickTuid = userDF[userDF.label == 0].drop_duplicates(["uid", "subeventid"])
    user_unclick_item_list = unDuplicationUnClickTuid.groupby("uid")["subeventid"].apply(list).to_frame().to_dict()
    user_unclick_item_dict = user_unclick_item_list['subeventid']

    # candHistoryRowDF = userDF[["uid", "candidate_set", "label"]].explode("candidate_set")
    #
    # candHistoryRowDF = candHistoryRowDF.reset_index(drop=True)
    # candHistoryRowDF.rename(columns={"candidate_set": "candidate_id"}, inplace=True)
    # candHistoryRowDF['candidate_id'] = candHistoryRowDF['candidate_id'].astype('int64')
    #
    # # 显示所有列
    # pd.set_option('display.max_columns', None)
    # userDF = pd.merge(userDF, candHistoryRowDF, how='left', left_on=['uid', 'label'], right_on=['uid', 'label'])
    #
    # 被推荐侧数据
    itemDF = pd.read_csv("C:/Users/EDZ/Desktop/item.csv")

    # 按照性别区分的候选集
    sexCandidateDict = dict()
    # 男生用户候选集
    # maleCandidateSet = itemDF[itemDF.tsex == 'm']
    sexCandidateDict['m'] = itemDF[itemDF.tsex == 'f']

    # 女生用户候选集
    # femaleCandidateSet = itemDF[item.tsex == 'f']
    sexCandidateDict['f'] = itemDF[itemDF.tsex == 'm']

    train_set, test_set = gen_data_set(userDF, itemDF, sexCandidateDict, negsample=0)



    # 性别不同的候选集
    # data = pd.merge(userDF, itemDF, how='left', left_on=['candidate_id'], right_on=['tuid'])
    # fillValue = {"tuid": -1, "tage": -1, "tworkid": -1, "theight": -1}
    # data = data.fillna(fillValue)
    # data['tuid'] = data['tuid'].astype('int64')
    # data['tage'] = data['tage'].astype('int64')
    # data['tworkid'] = data['tworkid'].astype('int64')
    # data['theight'] = data['theight'].astype('int64')
    #
    # print(data.head())




    # userDF = userDF.reset_index(drop=True)
    # # 正样本数据集
    # pos_sampes = userDF[userDF.label == 1]
