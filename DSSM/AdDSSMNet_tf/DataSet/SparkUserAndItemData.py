import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, IntegerType
from pyspark import StorageLevel
from itertools import chain
from pyspark.ml.feature import StringIndexer, FeatureHasher


schema = StructType([
    StructField("uid", StringType(), True),
    StructField("age", StringType(), True),
    StructField("workid", StringType(), True),
    StructField("height", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("timeStamp", StringType(), True),
    StructField("tuid", StringType(), True),
    StructField("tage", StringType(), True),
    StructField("tworkid", StringType(), True),
    StructField("theight", StringType(), True),
    StructField("tsex", StringType(), True),
    StructField("label", StringType(), True)
])


def someColumnRenamed(df, df1):
    df = df.select("uidIndex", "ageIndex", "workidIndex", "heightIndex", "sexIndex", "subeventidIndex", "timeStamp", "label")
    df = df.withColumnRenamed("uidIndex", "uid")
    df = df.withColumnRenamed("ageIndex", "age")
    df = df.withColumnRenamed("workidIndex", "workid")
    df = df.withColumnRenamed("heightIndex", "height")
    df = df.withColumnRenamed("sexIndex", "sex")
    df = df.withColumnRenamed("subeventidIndex", "subeventid")
    df1 = df1.select("tuidIndex", "tageIndex", "tworkidIndex", "theightIndex", "tsexIndex", "timeStamp1")
    df1 = df1.withColumnRenamed("tuidIndex", "tuid")
    df1 = df1.withColumnRenamed("tageIndex", "tage")
    df1 = df1.withColumnRenamed("tworkidIndex", "tworkid")
    df1 = df1.withColumnRenamed("theightIndex", "theight")
    df1 = df1.withColumnRenamed("tsexIndex", "tsex")
    return df, df1


def some_last_days(days):
    today = datetime.datetime.now()
    offset = datetime.timedelta(days=-days)
    beforeDate = (today + offset).strftime('%Y-%m-%d')
    return beforeDate


UserSql = """
    select
        t1.uid as uid,
        t1.age as age,
        t1.work_location as workid,
        t1.height as height,
        t1.sex as sex,
        t2.subeventid as subeventid,
        t2.timeStamp as timeStamp,
        if(t2.eventid == "8.32", 0, 1) as label
    from
        (
            select
                cast(uid as string) uid,
                cast(year(CURRENT_DATE) as int) - cast(birth_year as int) as age,
                work_location,
                height,
                sex
            from
                algorithm.users_search
            where
                gid >= 3
                and gid <= 10
                and uid is not null
                and work_location = "51"
         )t1
    inner join
        (
             select
                 t.uid,
                 t.subeventid,
                 t.`time` as timeStamp,
                 t.eventid
             from
                 algorithm.dataplatform_user_action_record t
             where
                 dt >= '{}'
                 and dt < '{}'
                 and eventid in ("8.32", "8.33", "8.34")
                 and subeventid is not null
                 and length(subeventid) > 0 
        )t2
    on t1.uid = t2.uid
""".format(some_last_days(1), some_last_days(0))

ItemSql = """
    select
        t1.uid as tuid,
        t1.age as tage,
        t1.work_location as tworkid,
        t1.height as theight,
        t1.sex as tsex,
        t2.timeStamp as timeStamp1
    from
        (
            select
                cast(uid as string) uid,
                cast(year(CURRENT_DATE) as int) - cast(birth_year as int) as age,
                work_location,
                height,
                sex
            from
                algorithm.users_search
            where
                gid >= 3
                and gid <= 10
                and uid is not null
                and work_location = "51"
            group by
                uid, age, work_location, height, sex
         )t1
    left join
        (
            select
                uid,
                timeStamp
            from
                (
                    select
                        uid,
                        `time` as timeStamp,
                        ROW_NUMBER() OVER(PARTITION BY uid ORDER BY `time` desc) AS rn
                    from
                        algorithm.dataplatform_user_action_record
                    where
                        dt >= "{}"
                        and dt < "{}"
                )t
            where
                t.rn = 1
        )t2
    on
        cast(t1.uid as string) = t2.uid
""".format(some_last_days(1), some_last_days(0))

uid2id = """
    select
        uid,
        id
    from
        algorithm.uid_convert_to_id
"""


# 数据处理部分
class SparkUserAndItemData(object):
    '''
    双塔模型召回侧数据加工
    '''

    def __init__(self, outfile):
        self.spark = SparkSession.builder.appName("sparkUserAndItemData").enableHiveSupport().getOrCreate()
        # self.spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        # self.spark.conf.set("spark.sql.crossJoin.enabled", "true")
        self.out_files = outfile
        self.age_dict = {i: str(i + 1) for i in range(100)}
        self.workid_dict = {i: str(i + 1) for i in range(100)}
        self.sex_dict = {'f': '1', 'm': '2', '-1': '0'}
        self.height_dict = {i: str(i - 99) for i in range(100, 227)}
        # 添加有效类别的字典
        # self.user_data = self.spark.sql(UserSql).persist(StorageLevel(True, True, False, False, 1))
        # DISK_ONLY = StorageLevel(True, False, False, False, 1)
        # DISK_AND_ME
        self.user_data, self.item_data = self.feature_convert_id()
        self.user_data.persist(StorageLevel(True, True, False, False, 1))
        self.item_data.persist(StorageLevel(True, False, False, False, 1))


    def feature_convert_id(self):
        user_data = self.spark.sql(UserSql)
        item_data = self.spark.sql(ItemSql)
        uid_id = self.spark.sql(uid2id)
        user_data = user_data.join(uid_id, ['uid'], "inner").withColumnRenamed("id", "uidIndex")
        user_data = uid_id.join(user_data, uid_id.uid == user_data.subeventid, "inner").drop(uid_id.uid).withColumnRenamed("id", "subeventidIndex")
        item_data = item_data.join(uid_id, uid_id.uid == item_data.tuid, "inner").drop(uid_id.uid).withColumnRenamed("id", "tuidIndex")
        # for feature in ["uid", "age", "workid", "height", "sex"]:
        #     user_data, item_data = labelEncoderExample(user_data, item_data, feature)
        # user_data, item_data = labelEncoderExample(user_data, item_data, "uid")
        # uid特征哈希处理
        # user_data = user_data.withColumn("uidIndex", F.col("uid").cast(IntegerType()) % 500000)
        # user_data = user_data.withColumn("subeventidIndex", F.col("subeventid").cast(IntegerType()) % 500000)
        # item_data = item_data.withColumn("tuidIndex", F.col("tuid").cast(IntegerType()) % 500000)
        # 年龄转化
        age_mapping_expr = F.create_map([F.lit(x) for x in chain(*self.age_dict.items())])
        user_data = user_data.withColumn("ageIndex", age_mapping_expr.getItem(F.col("age")))
        item_data = item_data.withColumn("tageIndex", age_mapping_expr.getItem(F.col("tage")))
        # 身高转化
        height_mapping_expr = F.create_map([F.lit(x) for x in chain(*self.height_dict.items())])
        user_data = user_data.withColumn("heightIndex", height_mapping_expr.getItem(F.col("height")))
        item_data = item_data.withColumn("theightIndex", height_mapping_expr.getItem(F.col("theight")))
        # 工作地点转化
        workid_mapping_expr = F.create_map([F.lit(x) for x in chain(*self.workid_dict.items())])
        user_data = user_data.withColumn("workidIndex", workid_mapping_expr.getItem(F.col("workid")))
        item_data = item_data.withColumn("tworkidIndex", workid_mapping_expr.getItem(F.col("tworkid")))
        # 性别转化
        sex_mapping_expr = F.create_map([F.lit(x) for x in chain(*self.sex_dict.items())])
        user_data = user_data.withColumn("sexIndex", sex_mapping_expr.getItem(F.col("sex")))
        item_data = item_data.withColumn("tsexIndex", sex_mapping_expr.getItem(F.col("tsex")))

        user_data, item_data = someColumnRenamed(user_data, item_data)
        return user_data, item_data

    def write_tfrecord_file(self, df):
        # df.repartition(10).write.format("tfrecords").option("recordType", "Example").mode("overwrite").save(
        #     self.out_files)
        df.write.format("csv").option("header", "true").mode("overwrite").save(
            "/user/algorithm/TwoTowerDataSet/")

    def processing_data_set(self):

        # 挑选出点击和未点击用户
        # click_user = self.user_data.filter(F.col("label") == 1)
        # unclick_user = self.user_data.filter(F.col("label") == 0)
        # w = Window.partitionBy("uid").orderBy("timeStamp")

        # 获取被推荐人的属性特征
        sample_data = self.user_data.join(self.item_data, self.user_data.subeventid == self.item_data.tuid, how='left').\
            select("uid", "age", "workid", "height", "sex", F.when(F.col("timeStamp").isNotNull(), F.col("timeStamp")).otherwise(F.col("timeStamp1")).alias("timeStamp"), "tuid", "tage", "tworkid", "theight", "tsex", "label")

        # 排除外省数据
        sample_data = sample_data.filter(F.col("tuid").isNotNull())

        # 获取60天的活跃用户
        activate_user_data = self.item_data.filter(F.col("timeStamp1").isNotNull())

        # 获取性别不同的用户
        sex_activate_user = activate_user_data.groupby("tsex").agg(F.collect_set("tuid").alias("tuids"))

        # 活跃用户转化成pandas.DataFrame
        sex_activate_user_dict = sex_activate_user.toPandas().set_index('tsex').to_dict()

        # bc_sample_data = self.spark.sparkContext.broadcast(activate_user_data)

        # @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        # # Input/output are both a pandas.DataFrame
        # def neg_sample(x):
        #     # 展开浏览过的历史
        #     # x = x.select(x.uid, x.age, x.workid, x.height, x.sex, F.explode(x.hist_tuid).alias("tuid"))
        #     x = x.explode("hist_tuid").reset_index(replace=True)
        #     # 删除性别相同的
        #     # sex_df = x.join(broadcast(activate_user_data), x.sex == activate_user_data.tsex, "leftanti")
        #     sex_df = x.join(activate_user_data_convert, x.sex == activate_user_data_convert.tsex, "leftanti")
        #     # 删除浏览历史中相同的
        #     # select_unhist_user = sex_df.join(x, x["tuid"] == sex_df["tuid"].cast(StringType()), how='leftanti')
        #     select_unhist_user = sex_df.join(x, x["tuid"] == sex_df["tuid"].astype("str"), how='leftanti')
        #     # 随机抽样
        #     # random_sample = select_unhist_user.sample(False, 0.05, seed=2021)
        #     random_sample = select_unhist_user.sample(frac=0.05, replace=False)
        #
        #     # 新增一列过度列
        #     x = x.withColumn("match_sex", F.when(F.col("sex") == "f", "m").otherwise("f"))
        #     random_sample = random_sample.join(x, x.match_sex == random_sample.tsex, how="left")
        #     random_sample = random_sample.select("uid", "age", "workid", "height", "sex", "timeStamp", "tuid", "tage",
        #                                          "tworkid", "theight", "tsex")
        #     # random_sample = random_sample.withColumn("label", F.lit(0))
        #     random_sample["label"] = [0]*random_sample.shape[0]
        #
        #     return random_sample

        # 添加新的一列，用户表示负采样的过程
        def neg_sample(sex, hist_tuid):
            hist_tuids = hist_tuid
            sex = '1' if sex == '2' else '1'
            candidate_user = sex_activate_user_dict['tuids'][sex]
            rest_tuids = [i for i in candidate_user if i not in hist_tuids]
            import random
            return random.sample(rest_tuids, 50)

        func = F.udf(neg_sample, ArrayType(StringType()))

        user_hist_data = self.user_data.groupby("uid", "age", "workid", "height", "sex").agg(F.collect_set("subeventid").alias("hist_tuid"))
        user_hist_data = user_hist_data.withColumn("neg_sample", func(user_hist_data.sex, user_hist_data.hist_tuid))
        # 返回数据结果
        # 集合浏览历史记录
        # user_hist_data = self.user_data.groupby("uid", "age", "workid", "height", "sex").agg(
        #     F.collect_set("subeventid").alias("hist_tuid")).groupby("uid", "age", "workid", "height", "sex", "hist_tuid").apply(neg_sample)

        # 已有数据和随机采样数据组合
        # total_data = sample_data.unionAll(user_hist_data)
        neg_samples = user_hist_data.select("uid", "age", "workid", "height", "sex", F.explode("neg_sample").alias("tuid"))
        neg_samples = neg_samples.withColumn("label", F.lit(0))

        activate_user_data1 = activate_user_data.withColumnRenamed("timeStamp1", "timeStamp")
        neg_samples = neg_samples.join(activate_user_data1, ["tuid"], how="left").select("uid", "age", "workid", "height", "sex", "timeStamp", "tuid", "tage", "tworkid", "theight", "tsex", "label")

        # 添加行
        sample_data = sample_data.unionAll(neg_samples)

        # 删除重复的数据
        # window = Window.partitionBy(['label']).orderBy(F.col('label').desc())
        # sample_data = sample_data.withColumn('rank', F.rank().over(window)).filter("rank= '1'").drop('rank')
        sample_data = sample_data.sort(F.col("label").desc()).dropDuplicates(["uid", "tuid"])

        # 筛选出点击的用户
        click_user = sample_data.filter(F.col("label") == 1).select("uid", "tuid", "timeStamp").sort(F.col("timeStamp"))
        collect_user_data = click_user.groupby("uid").agg(F.collect_list("tuid").alias("tuids"), F.collect_list("timeStamp").alias("timeStamps"))

        # 加工历史数据
        def combine_time_hist_tuid(timeStamp, hist_tuid):
            result = {}
            for i in range(1, len(timeStamp)):
                if i > 40:
                    result[timeStamp[i - 1] + "_" + timeStamp[i]] = hist_tuid[i-40:i]
                else:
                    result[timeStamp[i - 1] + "_" + timeStamp[i]] = hist_tuid[:i]
            return result

        func1 = F.udf(combine_time_hist_tuid, MapType(StringType(), ArrayType(StringType())))
        time_hist_tuid_data = collect_user_data.withColumn("conbime_time_tuid", func1(collect_user_data.timeStamps, collect_user_data.tuids)).select("uid", "conbime_time_tuid")

        # 用户时间进行关联
        uid_data_change = sample_data.join(time_hist_tuid_data, ["uid"], how="left")

        # 过滤历史数据
        def filter_hist_tuid(timeStamp, hist_tuid):
            if not hist_tuid:
                return []
            for key in hist_tuid:
                start_time, end_time = key.split("_")
                if start_time < timeStamp <= end_time:
                    return hist_tuid[key]
            return []

        func2 = F.udf(filter_hist_tuid, ArrayType(StringType()))
        # 新增一列用户历史浏览记录
        uid_data1 = uid_data_change.withColumn("watched_history_uid", func2(uid_data_change.timeStamp, uid_data_change.conbime_time_tuid))
        uid_data1 = uid_data1.drop("conbime_time_tuid").filter(F.col("watched_history_uid").isNotNull())
        uid_data1 = uid_data1.select("uid", "age", "workid", "height", "sex", "timeStamp", "watched_history_uid", "tuid", "tage", "tworkid", "theight", "tsex", "label")
        # # 浏览过的人往下移动一行
        # uid_data_change = click_user.withColumn("change_uid_hist", F.lag("tuid").over(Window.partitionBy("uid").orderBy("timeStamp1")))
        # # 删除带控制的行
        # uid_data = uid_data_change.filter(F.col("change_uid_hist").isNotNull() | F.col("change_uid_hist") != '')
        # # 新增一列用户历史浏览记录
        # uid_data1 = uid_data.withColumn("watched_history_uid", F.collect_set("change_uid_hist").over(Window.partitionBy("uid").orderBy("timeStamp1")))
        #
        # # 增加数据
        # sample_data.join(uid_data1, ["uid"], how="left").filter(F.col("time"))

        # 首先大量稀疏数据labelEncoder编码
        # indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(uid_data1) for column in list(uid_data1.columns)]
        #
        # pipeline = Pipeline(stages=indexers)
        # uid_data1 = pipeline.fit(uid_data1).transform(uid_data1)

        def array_to_string(my_list):
            return '&'.join([str(elem) for elem in my_list])

        array_to_string_udf = F.udf(array_to_string, StringType())
        uid_data1 = uid_data1.withColumn('watched_history_uid_list', array_to_string_udf(uid_data1["watched_history_uid"])).drop("watched_history_uid").filter(F.col("watched_history_uid_list") != '')

        self.write_tfrecord_file(uid_data1)

        self.close()

    def close(self):
        self.spark.stop()


if __name__ == "__main__":
    sparkUserAndItemData = SparkUserAndItemData("/user/algorithm/TwoTowerDataSet/")
    sparkUserAndItemData.processing_data_set()