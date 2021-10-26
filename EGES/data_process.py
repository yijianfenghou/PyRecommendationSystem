import pandas as pd
import numpy as np
from itertools import chain
import pickle as pk
import random
import time
import networkx as nx
import os, sys
path = os.path.dirname(__file__)
sys.path.append(path)

from walker import RandomWalker

from sklearn.preprocessing import LabelEncoder
import argparse


# 根据frequency设定one-hot id
def process_id(graph):
    index = 0
    mid_to_id = {}
    id_to_mid = {}
    for key, idx in sorted(graph.mid_freq.items(), key=lambda kv:kv[1]):
        mid_to_id[key] = index
        id_to_mid[index] = key
        index += 1
    return mid_to_id, id_to_mid


# side information的one-hot操作
def feature_process(fea_list):
    one_hot_fea = []
    age, edu, sal, height, mar = fea_list
    # age [0-7]
    if age == '0' or age == '-1':
        one_hot_fea.append(0)
    elif age <= '25':
        one_hot_fea.append(1)
    elif age <= '30':
        one_hot_fea.append(2)
    elif age <= '35':
        one_hot_fea.append(3)
    elif age <= '40':
        one_hot_fea.append(4)
    elif age <= '50':
        one_hot_fea.append(5)
    elif age <= '60':
        one_hot_fea.append(6)
    else:
        one_hot_fea.append(7)

    # edu [0, 2-7]
    one_hot_fea.append(0 if edu == '0' or edu == '-1' else int(edu)-1)

    # sal [0, 3-9]
    one_hot_fea.append(0 if sal == '0' or sal == '-1' else int(sal)-2)

    # height [0-11]
    if height == '0' or height =='-1':
        one_hot_fea.append(0)
    elif height <= '150':
        one_hot_fea.append(1)
    elif height <= '155':
        one_hot_fea.append(2)
    elif height <= '160':
        one_hot_fea.append(3)
    elif height <= '165':
        one_hot_fea.append(4)
    elif height <= '170':
        one_hot_fea.append(5)
    elif height <= '175':
        one_hot_fea.append(6)
    elif height <= '180':
        one_hot_fea.append(7)
    elif height <= '185':
        one_hot_fea.append(8)
    elif height <= '190':
        one_hot_fea.append(9)
    else:
        one_hot_fea.append(10)

    # mar [0, 1, 3, 4]
    one_hot_fea.append(int(mar) if mar in ('0', '1') else int(mar)-1)

    return one_hot_fea


# 生成training batch
def generate_batch(graph, info_dict, batch_size, num_paths=2, path_length=8, window_size=2, neg_num=0):
    xs = []
    ys = []

    print("开始deep walk...")
    # generate deep walk corpus
    walk_corpus = graph.build_deepwalk_corpus(num_paths, path_length)

    print('根据frequency生成新的id...')
    # generate ont-hot index based on nodes' frequency
    mid_to_id = process_id(graph)[0]

    print('负采样的所需的频率list...')
    # generate negative sampling list based on frequency, default M=10**8
    ns_list = ng_data(graph)

    print('生成skip_gram训练数据...')
    # select a walk list
    for i in range(len(walk_corpus)):
        walk = walk_corpus[i]
        walk_length = len(walk)
        # select target word
        for j in range(walk_length):
            for k in range(max(0,j-window_size), min(walk_length,j+window_size)):
                if j == k:
                    continue

                xs.append([mid_to_id[walk[j]]] + info_dict[walk[j]])
                ys.append(mid_to_id[walk[k]])
                if neg_num>0:
                    ns_list = sample_k_neg(ns_list, walk[j], neg_num)
                    for ns_id in ns_list:
                        xs.append(mid_to_id[walk[j]] + info_dict[walk[j]])
                        ys.append(mid_to_id[ns_id])

    batch_x = []
    batch_y = []
    for i in range(len(xs)//batch_size):
        batch_x.append(xs[i*batch_size: (i+1)*batch_size])
        batch_y.append(ys[i*batch_size: (i+1)*batch_size])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    print('shape of batch_x:', batch_x.shape)
    print('shape of batch_y:', batch_y.shape)
    return batch_x, batch_y


def ng_data(graph, M=10**8):
    res = graph.mid_freq

    count_sum = 0
    for mid,count in res.items():
        count_sum += count**0.75

    ns_list = np.array([0 for i in range(M)])
    cur_pro = 0
    index = 0
    for i,mid in enumerate(graph.G.nodes()):
        cur_pro = min(cur_pro+(res[mid]**0.75/count_sum), 1)   # 防止运算出现误差带来的错误
        while i < M and index/M < cur_pro:
            ns_list[index] = mid
            index += 1
    return ns_list


def sample_k_neg(ns_list, target, k):
    ns = []
    for i in range(k):
        neg_index = ns_list[random.randint(0, ns_list.size)]
        while neg_index not in ns and neg_index != target:
            ns.append(neg_index)
    return ns


def process_data(file_name, batch_size, num_paths, path_length, window_size):
    # 生成session文件
    generate_sess(file_name)

    with open('session.pkl', 'rb') as f:
        data = pk.load(f)
    with open('info.pkl', 'rb') as f:
        info_dict = pk.load(f)

    '''
    查看生成数据信息
    '''
    rec = {}
    print('length of session_list:', len(data))

    # walks的结果
    s = 0
    for d in data:
        s += len(d[1])
        if d[0] not in rec:
            rec[d[0]] = 0
        for oid in d[1]:
            if oid not in rec:
                rec[oid] = 0
    print('number of walks:', s)
    print('number of nodes in walks:', len(rec.keys()))

    # 查看info_dict中包含了多少个id的特征
    print('number of id in info_dict:', len(info_dict.keys()))

    deep_walk_graph = Graph(data)
    mid_to_id, id_to_mid = process_id(deep_walk_graph)
    # print(mid_to_id)
    batch_x, batch_y = generate_batch(deep_walk_graph, info_dict, batch_size=100, num_paths=2, path_length=8, window_size=4, neg_num=0)

    with open('batch_x.pkl', 'wb') as f:
        pk.dump(batch_x, f)
    with open('batch_y.pkl', 'wb') as f:
        pk.dump(batch_y, f)
    with open('mid_to_id.pkl', 'wb') as f:
        pk.dump(mid_to_id, f)
    with open('id_to_mid', 'wb') as f:
        pk.dump(id_to_mid, f)


def cut_session(data, time_cut=1):

    cust_id_list = data["cust_id"]
    time_list = data["time"]

    session = []
    tmp_session = []
    for i, item in enumerate(cust_id_list):
        if (i < len(cust_id_list)-1 and (time_list[i+1] - time_list[i]).days > time_cut) or i == len(cust_id_list) - 1:
            tmp_session.append(item)
            session.append(tmp_session)
            tmp_session = []
        else:
            tmp_session.append(item)
    return session


def get_session(action_data, use_type=None):

    # action_data = action_data[action_data['type'].isin(use_type)]
    action_data = action_data.sort_values(by=['emp_id', 'operate_time'], ascending=True)
    group_action_data = action_data.groupby('emp_id').agg(list)
    session_list = group_action_data.apply(cut_session, axis=1)
    return session_list.to_numpy()


def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i-window_size, i+window_size):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])

    return pd.DataFrame(all_pairs, columns=["cust_id", "candidate_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--q", type=float, default=2)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_known_args()[0]

    # action_data = pd.read_csv(args.data_path+"action_head.csv", parse_dates=['action_time']).drop('module_id', axis=1).dropna()
    action_data = pd.read_csv(args.data_path + "t1.csv")

    action_data['time'] = action_data['operate_time'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    action_data['time'] = pd.to_datetime(action_data['time'])

    all_custs = action_data["cust_id"].unique()
    all_custs = pd.DataFrame({"cust_id": list(all_custs)})
    custs_lbe = LabelEncoder()
    all_custs['cust_id'] = custs_lbe.fit_transform(all_custs['cust_id'])
    action_data['cust_id'] = custs_lbe.transform(action_data['cust_id'])

    # make session list
    start_time = time.time()
    session_list = get_session(action_data)

    session_list_all = []
    for item_list in session_list:
        for session in item_list:
            if len(session) > 1:
                session_list_all.append(session)

    # session2graph
    node_pair = dict()
    for session in session_list_all:
        for i in range(1, len(session)):
            if (session[i-1], session[i]) not in node_pair:
                node_pair[(session[i-1], session[i])] = 1
            else:
                node_pair[(session[i-1], session[i])] += 1

    in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
    out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
    weight_list = list(node_pair.values())
    graph_df = pd.DataFrame({'in_node': in_node_list, 'out_node': out_node_list, 'weight': weight_list})
    graph_df.to_csv('./data_cache/graph.csv', sep=' ', index=False, header=False)

    G = nx.read_edgelist('./data_cache/graph.csv', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    walker = RandomWalker(G, p=args.p, q=args.q)
    print("Preprocess transition probs...")
    walker.preprocess_transition_probs()

    session_reproduce = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, verbose=1)
    session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    # add side info
    cust_side_info = action_data[['cust_id', 'income', 'education', 'age']].drop_duplicates("cust_id")

    # all_custs['cust_id'] = custs_lbe.inverse_transform(all_custs['cust_id'])
    # cust_side_info = pd.merge(all_custs, product_data, on='sku_id', how='left').fillna(0)

    # id2index
    for feat in cust_side_info.columns:
        if feat != 'cust_id':
            lbe = LabelEncoder()
            cust_side_info[feat] = lbe.fit_transform(cust_side_info[feat])
        # else:
        #     cust_side_info[feat] = custs_lbe.transform(cust_side_info[feat])

    cust_side_info = cust_side_info.sort_values(by=['cust_id'], ascending=True)
    cust_side_info.to_csv('./data_cache/cust_side_info.csv', index=False, header=False, sep='\t')

    # get pair
    all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
    all_pairs["cust_id"] = all_pairs["cust_id"].astype("int64")
    all_pairs["candidate_id"] = all_pairs["candidate_id"].astype("int64")

    # add side infomation
    a = all_pairs["cust_id"].values
    b = cust_side_info["cust_id"].values
    all_pairs_side_info = cust_side_info.merge(all_pairs, on="cust_id", how="left")
    all_pairs_side_info.dropna(subset=['candidate_id'], inplace=True)
    np.savetxt('./data_cache/all_pairs', X=all_pairs_side_info, fmt="%d", delimiter=" ")
