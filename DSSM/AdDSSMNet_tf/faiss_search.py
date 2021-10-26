import pandas as pd
import numpy as np
import faiss
# import json
import pickle
import redis
import datetime
# from DSSM.AdDSSMNet_tf.utils.mongo_uid_type import *

def some_last_days(days):
    today = datetime.datetime.now()
    offset = datetime.timedelta(days=-days)
    beforeDate = (today + offset).strftime('%Y-%m-%d')
    return beforeDate

getActiveUsers = """
    select
        /*+ BROADCAST(dataplatform_user_action_record)*/
        t1.uid,
        t1.age,
        case when t1.match_min_age is null and t1.sex = 'f' then t1.age when t1.match_min_age is null and t1.sex = 'm' then 17 else t1.match_min_age end as match_min_age,
        case when t1.match_max_age is null and t1.sex = 'f' then t1.age+10 when t1.match_max_age is null and t1.sex = 'm' then t1.age+2 else t1.match_max_age end as match_max_age,
        t1.work_location,
        t1.sex,
        nvl(t5.type, "2") type
    from
        (
            select
                uid,
                cast(year(now()) as int) - cast(birth_year as int) age,
                match_min_age,
                match_max_age,
                work_location,
                sex
            from
                algorithm.users_search
            where
                gid >= 3.0
                and gid <= 10.0
                and uid is not NULL
                and work_location = 51
         )t1
    inner join
        (
             select
                 uid
             from
                algorithm.dataplatform_user_action_record
             where
                dt >= "%s"
                and eventid in ('8.32', '8.33', '8.34')
                and subeventid is not NULL
                and subeventid != ''
             group by
                uid
        )t2
        on cast(t1.uid as string) = t2.uid
    left join
        (
            select
                user_id,
                service_id,
                type
            from
                (
                    select
                        user_id,
                        service_id,
                        type,
                        row_number() over(partition by user_id order by type) num
                    from
                    (
                        select
                            user_id,
                            service_id,
                            case when service_id = '99' and times = 0 then '2' else type end type
                        from
                        (
                            select
                                user_id,
                                trim(service_id) service_id,
                                times
                            from
                               algorithm.jy_user_service
                        )t3
                        left join
                        (
                            select
                                id,
                                trim(type) type
                            from
                                algorithm.jy_user_service_type
                        )t4
                        on cast(t3.service_id as string) = t4.id
                )tmp1
           )tmp2
           where
               tmp2.num = 1
        )t5
    on cast(t1.uid as string) = cast(t5.user_id as string)
""".format(some_last_days(90))


class FaissCB:

    def __init__(self, k):
        self.n_sim_item = k

    def fit(self, item_matrix, ids):
        num, vec_dim = item_matrix.shape
        # 创建索引
        self.faiss_index = faiss.IndexFlatL2(vec_dim)  # 使用欧式距离作为变量
        # 添加id编号
        self.faiss_index1 = faiss.IndexIDMap(self.faiss_index)
        # 添加数据
        self.faiss_index1.add_with_ids(item_matrix, ids)
        # self.faiss_index.add(item_matrix)

    def predict(self, user_matrix):
        res_distance, res_index = self.faiss_index1.search(user_matrix, self.n_sim_item)
        return res_distance, res_index


class WriteRecallRedis:

    def __init__(self, uids):
        self.uids = uids

    def write_data_redis(self):
        r = redis.StrictRedis(host='10.2.0.35', port=6379, db=0)
        recall_dict = {}
        for index, row in self.uids.iterrows():
            user_id = str(row["uid_type_workid"])
            recom_tids = str(row["uid"]).split(",")
            if user_id not in recall_dict:
                recall_dict[user_id] = {}

            if str(row["type"]) not in recall_dict[user_id]:
                recall_dict[user_id][str(row["type"])] = recom_tids

        for user in recall_dict:
            # 数据写入redis
            p1 = pickle.dumps(recall_dict[user])
            r.set(user, p1)
        r.close()


# 加工用户类型
# class UserType:
#
#     def __init__(self, env):
#         self.env = env
#         self.user_service_type_server = MongoServer(env=env, table='user_service_type_v1.2')
#
#     def getUsersTypeFromMongo(self, uids):
#
#         def trans(data):
#             uid = data['uid']
#             uid_type = data['service_type']
#             return uid, uid_type
#
#         def analyze_response(cursor, limit=1000):
#             res = []
#             i = 0
#             for each in cursor:
#                 i += 1
#                 res.append(each)
#                 if i > limit:
#                     raise Exception('cursor overflow limit 1000')
#             return res
#
#         # uid 数据类型转换 int -> str
#         assert isinstance(uids, list)
#         uids = list(map(lambda x: str(x), uids))
#         condition = {'uid': {'$in': uids}}
#         res = []
#         cursor = self.user_service_type_server.find(condition, {'_id': 0})
#         contents = analyze_response(cursor)
#         if len(contents) > 0:
#             res = list(map(trans, contents))
#         return res

# 加工用户类型以及匹配条件
class UserAndMatchCond:

    def __init__(self, host="10.1.1.244", port=21050, database="algorithm"):
        self.host = host
        self.port = port
        self.database = database

    def get_user_type_sex(self):
        from impala.dbapi import connect
        conn = connect(host=self.host, port=self.port, database=self.database)
        # 定点游标
        cursor = conn.cursor()
        hive_sentence = getActiveUsers
        cursor.execute(hive_sentence)
        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        df = pd.DataFrame(result)
        return df


if __name__ == "__main__":
    fcb = FaissCB(k=100)
    items = pd.read_csv("./output/result/tensorflow_item_embedding.csv")
    item_ids = items["uid"].values.astype(np.int64)

    # item_ids = []
    item_matrix = []
    item_index_mapping = {}  # {item_matrix_index: item_id}
    for feature in items["item_embedding"]:
        # print(item_id)
        # item_ids.append(item_id)
        feature = [float(i) for i in feature[1:-1].replace("\n", "").split(" ") if i != '']
        # feature = list(map(lambda x: float(x), feature[1:-1].replace("\n", "").split(" ")))
        item_matrix.append(feature)
        # item_index_mapping[index] = int(item_id)
        # index += 1

    # item_ids = np.array(item_ids, dtype="int64")
    item_matrix = np.array(item_matrix, dtype="float32")
    fcb.fit(item_matrix, item_ids)

    # 找出和用户最相似的k个推荐用户
    users = pd.read_csv("./output/result/tensorflow_user_embedding.csv")
    users = users.drop_duplicates(["uid"])
    # uids = users["user_id"].values.tolist()
    # uid = users[users["user_id"] == 10]["user_embedding"].iloc[0]
    # uid = list(map(lambda x: float(x), uid.split(",")))
    # user_embedding = np.array(uid, dtype="float32")
    # user_embedding = np.expand_dims(user_embedding, axis=0).astype(np.float32)
    uids = users["uid"].values.tolist()
    uids = uids[:2]

    uid_embedding = users["uid_embedding"].values.tolist()
    uid_embedding = [[float(i) for i in uid[1:-1].replace("\n", "").split(" ") if i != ''] for uid in uid_embedding]
    user_embedding = np.array(uid_embedding).astype(np.float32)[:2]

    res_distance, res_index = fcb.predict(user_embedding)

    result = pd.DataFrame(list(zip(uids, res_index)))
    result.columns = ["uid", "tuid"]
    result = result.explode("tuid").reset_index(drop=True)

    userAndMatchCond = UserAndMatchCond()
    uidDF = userAndMatchCond.get_user_type_sex()

    result = pd.merge(result, uidDF, how="left", on=["uid"])
    result.columns = ["user_id", "uid", "uid_age", "uid_match_min_age", "uid_match_max_age", "uid_work_location", "uid_sex", "uid_type"]

    result = pd.merge(result, uidDF, how="left", on=["uid"])
    result = result[["user_id", "uid", "uid_match_min_age", "uid_match_max_age", "uid_work_location", "uid_sex", "uid_type", "age", "sex", "type"]]

    result = result[(result.uid_match_min_age <= result.age) & (result.uid_match_max_age >= result.age)]
    result = result[result.uid_sex != result.sex]
    result = result[((result.uid_type == '0') & (result.type != '2')) | ((result.uid_type == '2') & (result.type != '0'))]
    result["uid_type_workid"] = result["user_id"].map(str) + "_" + result["uid_type"].map(str) + "_" + result["uid_work_location"].map(str)

    result["uid"] = result["uid"].astype("str")
    def ab(df):
        return ','.join(df.values)

    result = result.groupby(["uid_type_workid", "type"])["uid"].apply(ab).reset_index()
    # result.columns = ["uid_type_workid", "tuids"]

    writeRecallRedis = WriteRecallRedis(result)
    writeRecallRedis.write_data_redis()
    # 用户类型关联
    # uid_type = UserType("pe")
    #
    # uid_service_type = uid_type.getUsersTypeFromMongo(uids)
    # tuid_service_type = [uid_type.getUsersTypeFromMongo(list(tuids)) for tuids in res_index]

    # 获取的被推荐的ids
    # print(res_distance)
    # filterRules = FilterRules(uid_service_type, tuid_service_type)
    # filterRules.filter_type_unmatch_users()
