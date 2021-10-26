import os
import time
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
# from construct_tree import TreeInitialize

LOAD_DIR = os.path.dirname(os.path.abspath(__file__)) + "/datasets/UserBehavior_sp.csv"

def _time_window_stamp():
    boundaries = ['2017-11-26 00:00:00', '2017-11-27 00:00:00', '2017-11-28 00:00:00',
                  '2017-11-29 00:00:00', '2017-11-30 00:00:00', '2017-12-01 00:00:00',
                  '2017-12-02 00:00:00', '2017-12-03 00:00:00', '2017-12-04 00:00:00']
    for i in range(len(boundaries)):
        time_array = time.strptime(boundaries[i], "%Y-%m-%d %H:%M:%S")
        time_stamp = int(time.mktime(time_array))
        boundaries[i] = time_stamp
    return boundaries

def _time_converter(x, boundaries):
    tag = -1
    if x > boundaries[-1]:
        tag = 9
    else:
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
                tag = i
                break
    return tag

def _mask_padding(data, max_len):
    size = data.shape[0]
    raw = data.values
    mask = np.array([[-2]*max_len for _ in range(size)])
    for i in range(size):
        mask[i, :len(raw[i])] = raw[i]
    return mask.tolist()

def data_process():
    """
    convert and split the raw data.
    :return:
    """
    data_raw = pd.read_csv(LOAD_DIR, header=None, names=['user_ID', 'item_ID', 'category_ID', 'behavior_type', 'timestamp'])

    data_raw = data_raw[:10000]
    user_list = data_raw.user_ID.drop_duplicates().to_list()
    user_dict = dict(zip(user_list, range(len(user_list))))
    data_raw['user_ID'] = data_raw.user_ID.apply(lambda x: user_dict[x])

    item_list = data_raw.item_ID.drop_duplicates().to_list()
    item_dict = dict(zip(item_list, range(len(item_list))))
    data_raw['item_ID'] = data_raw.item_ID.apply(lambda x: item_dict[x])

    category_list = data_raw.category_ID.drop_duplicates().to_list()
    category_dict = dict(zip(category_list, range(len(category_list))))
    data_raw['category_ID'] = data_raw.category_ID.apply(lambda x: category_dict[x])

    behavior_dict = dict(zip(['pv', 'buy', 'cart', 'fav'], range(4)))
    data_raw['behavior_type'] = data_raw.behavior_type.apply(lambda x: behavior_dict[x])

    time_window = _time_window_stamp()
    data_raw['timestamp'] = data_raw.timestamp.apply(_time_converter, args=(time_window, ))

    random_tree = Tree
