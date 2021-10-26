import os, sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

import numpy as np
import faiss
import collections
from tqdm import tqdm
import pickle
import pandas as pd

from KaggleTopic.preProcessData import *


def embedding_sim(click_df, item_emb_df, save_path, topk):

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df.acticle_id))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_idx = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_idx.add(item_emb_np)

    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_idx.search(item_emb_np, topk)

    # 将原始检索的结果保存原始的id对应关系
    item_sim_dict = collections.defaultdict(dict)

    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        for rele_index, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_index]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + "emb_i2i_sim.pkl", "wb"))

    return item_sim_dict

if __name__ == "__main__":
    item_emb_df = pd.read_csv("../data/articles_emb.csv")
    save_path = "./temp_results/"
    all_click_df = get_all_click(offline=False)
    embedding_i2i_sim = embedding_sim(all_click_df, item_emb_df, save_path, topk=10)

