from deep_walk import Graph
import pickle as pk
import numpy as np
import random
import time


# 对比时间大小
def time_diff(t1, t2):
    if not t1 or not t2:
        return 0
    t1_timeStamp = int(time.mktime(time.strptime(t1, "%Y-%m-%d %H:%M:%S")))
    t2_timeStamp = int(time.mktime(time.strptime(t2, "%Y-%m-%d %H:%M:%S")))
    # 先判断大小
    return t1_timeStamp - t2_timeStamp


# 对源数据进行数据清洗并生成sessions和side information dict文件
def generate_sess(file_name):
    data = []
    info_data = {}
    with open(file_name, 'r') as f:
        cur_memberid = ''
        cur_objectid = ''
        cur_time = ''
        session = []  # 一小时的session
        for line in f.readlines():
            line_vec = line.split('\t')
            memberid, objectid, target, occur_time = line_vec[:4]
            m_info = line_vec[4:14]
            o_info = line_vec[14:-1]

            # 更新基础信息和择偶信息特征
            info_data[memberid] = feature_process(m_info[:5])
            info_data[objectid] = feature_process(o_info[:5])

            # 没有操作时间的跳过，或者时间有问题的跳过
            if occur_time == 'NULL' or int(occur_time.split('-')[0]) < 2020:
                continue

            # 当memberid不一样时，或者操作间隔超过一小时，则本次session结束，开始计算下一个session
            if memberid != cur_memberid or abs(time_diff(cur_time, occur_time)) > 3600:  # 3600*2: #  两小时一个session
                if len(session) > 2:  # 大于2的session才加入训练集
                    data.append([cur_memberid, session])
                # session清空，重新统计
                session = []

            # 判断mid&oid pair
            # 如果是同一对mo，只更新时间，不加入session
            if memberid == cur_memberid and objectid == cur_objectid:
                cur_time = occur_time
            # 不是同一对，则要更新所有信息
            else:
                # session.append([objectid,o_info])
                session.append(objectid)
                cur_memberid = memberid
                cur_objectid = objectid
                cur_time = occur_time
        if len(session) > 2:  # 大于2的session才加入训练集
            data.append([cur_memberid, session])

    with open('sessions.pkl', 'wb') as f:
        pk.dump(data, f)
    with open('info.pkl', 'wb') as f:
        pk.dump(info_data, f)


# 根据frequency设定one-hot id
def process_id(graph):
    index = 0
    mid_to_id = {}
    id_to_mid = {}
    for key, value in sorted(graph.mid_freq.items(), key=lambda kv: kv[1]):
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
    one_hot_fea.append(0 if edu == '0' or edu == '-1' else int(edu) - 1)

    # sal [0, 3-9]
    one_hot_fea.append(0 if sal == '0' or sal == '-1' else int(sal) - 2)

    # height [0-11]
    if height == '0' or height == '-1':
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

    # mar [0, 1 ,3 ,4]
    one_hot_fea.append(int(mar) if mar in ('0', '1') else int(mar) - 1)

    return one_hot_fea


# 生成training batch
def generate_batch(graph, info_dict, batch_size, num_paths=2, path_length=8, window_size=2, neg_num=0):
    xs = []
    ys = []

    print('开始deep walks...')
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
            for k in range(max(0, j - window_size), min(walk_length, j + window_size)):
                if j == k:
                    continue

                xs.append([mid_to_id[walk[j]]] + info_dict[walk[j]])
                ys.append(mid_to_id[walk[k]])
                if neg_num > 0:
                    ns_list = sample_k_neg(ns_list, walk[j], neg_num)
                    for ns_id in ns_list:
                        xs.append(mid_to_id[walk[j]] + info_dict[walk[j]])
                        ys.append(mid_to_id[ns_id])
    batch_x = []
    batch_y = []
    for i in range(len(xs) // batch_size):
        batch_x.append(xs[i * batch_size: i * batch_size + batch_size])
        batch_y.append(ys[i * batch_size: i * batch_size + batch_size])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    print('shape of batch_x:', batch_x.shape)
    print('shape of batch_y:', batch_y.shape)
    return batch_x, batch_y


# generate negative sampling list
def ng_data(graph, M=10 ** 8):
    res = graph.mid_freq

    count_sum = 0
    for mid, count in res.items():
        count_sum += count ** 0.75

    ns_list = np.array([0 for i in range(M)])
    cur_pro = 0
    index = 0
    for i, mid in enumerate(graph.G.nodes()):
        cur_pro = min(cur_pro + (res[mid] ** 0.75 / count_sum), 1)  # 防止运算出现误差带来的错误
        while i < M and index / M < cur_pro:
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

    with open('sessions.pkl', 'rb') as f:
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
    with open('id_to_mid.pkl', 'wb') as f:
        pk.dump(id_to_mid, f)


if __name__ == '__main__':
    process_data(file_name='sz_womem.txt')