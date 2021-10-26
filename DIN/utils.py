import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from collections import namedtuple, OrderedDict
from tensorflow.keras.preprocessing.sequence import pad_sequences

def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat_name': feat}

def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=4):
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i-1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    # feature columns
    feature_columns = [[], [sparseFeature('item_id', item_count, embed_dim)]]

    # behavior
    behavior_list = ['item_id']

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    train_X = [np.array([0.]*len(train)), np.array([0]*len(train)), pad_sequences(train['hist'], maxlen=maxlen), np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0.]*len(val)), np.array([0]*len(val)), pad_sequences(val['hist'], maxlen=maxlen), np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0.]*len(test)), np.array([0]*len(test)), pad_sequences(test['hist'], maxlen=maxlen), np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print(train_X)
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == "__main__":
    file = "raw_data/remap.pkl"
    maxlen = 20
    create_amazon_electronic_dataset(file, embed_dim=8, maxlen=maxlen)

