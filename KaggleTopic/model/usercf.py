import os, sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from KaggleTopic.preProcessData import *


def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    max_min_scaler = lambda x: (x - np.min(x))/(np.max(x)-np.min(x))
    all_click_df_['click_article_id'] = all_click_df_[['click_article_id']].apply(max_min_scaler)
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict

def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵

        思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    item_user_time_dict = get_item_user_time(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in item_user_time_dict.item():
        for u, u_click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, v_click_time in user_time_list:
                if u == v: continue
                u2u_sim[u].setdefault(v, 0)
                activate_weight = 100*0.5*(user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight/math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, releted_user in u2u_sim.items():
        for v, wij in releted_user:
            u2u_sim_[i][j] += wij / math.sqrt(user_cnt[u] * user_cnt[v])

    pickle.dump(u2u_sim_, open(save_path+"usercf_u2u_sim.pkl"), 'wb')
    return u2u_sim_
