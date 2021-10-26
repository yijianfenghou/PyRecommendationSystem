import pandas as pd
import numpy as np
import pickle
import tqdm as tqdm
from collections import defaultdict


def get_all_click_sample(data_path, sample_num=100000):
    """
    训练数中采样一部分数据调试
    :param data_path: 原始数据储存位置
    :param sample_num: 采样数据数目
    :return:
    """
    all_click = pd.read_csv(data_path, "train_click_log.csv")
    all_user_ids = all_click["user_id"].unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_num, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click

def get_all_click(data_path, offline=True):
    if offline:
        all_click = pd.read_csv(data_path + "train_click_log.csv")
    else:
        trn_click = pd.read_csv(data_path + "train_click_log.csv")
        tst_click = pd.read_csv(data_path + "testA_click_log.csv")

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click

def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + "articles.csv")

    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    return item_info_df

def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(data_path + "articles_emb.csv")

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])

    # 归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + "item_content_emb.pkl"), "wb")

    return item_emb_dict

def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby("user_id")['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict

def get_item_user_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(lambda x: make_user_time_pair(x)).reset_index().rename(names={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df

def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_create_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['create_at_ts']))

    return item_type_dict, item_words_dict, item_create_time_dict

def get_user_hist_item_info_dict(all_click):

    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_type = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_type_dict = dict(zip(user_hist_item_type['user_id'], user_hist_item_type['category_id']))

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_id = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_id_dict = dict(zip(user_hist_item_id['user_id'], user_hist_item_id['click_article_id']))

    # 获取user_id对应的用户历史点击的文章平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg(mean).reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sorted_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], user_last_item_created_time['created_at_ts']))

    return user_hist_item_type_dict, user_hist_item_id_dict,  user_hist_item_words_dict, user_last_item_created_time_dict

def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


if __name__ == "__main__":

    data_path = "./data/"
    save_path = "./temp_results/"
    metric_recall = False

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 全量训练集
    all_click_df = get_all_click(offline=False)

    all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

    item_info_df = get_item_info_df(data_path)
    item_emb_info = get_item_emb_dict(data_path)

    # 获取文章的属性信息，宝存成字典的形式方便查询
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

    # 定义一个多路召回的字典， 将各路召回的结果都保存在这个字典当中
    user_multi_recall_dict = {
        'itemcf_sim_itemcf_recall': {},
        'embedding_sim_item_recall': {},
        'youtubednn_recall': {},
        'youtubednn_usercf_recall': {},
        'cold_start_recall': {}
    }

    # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
    # 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
