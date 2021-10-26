import os, sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from KaggleTopic.preProcessData import *


def itemcf(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵

        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in user_item_time_dict:
        # 在基于商品的协同过滤游湖的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # 考虑文章的正向顺序点击和反向顺序点击
                local_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = local_alpha * (0.9**(np.abs(loc2-loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7**np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8**np.abs(item_created_time_dict))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终文章之间的相似度
                i2i_sim[i][j] += loc_weight*click_time_weight*created_time_weight/math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij/math.sqrt(item_cnt[i]*item_cnt[j])

    # 将得到的相似度矩阵保存在本地
    pickle.dump(i2i_sim_, open(save_path+"itemcf_i2i_sim.pkl", "wb"))

    return i2i_sim_



