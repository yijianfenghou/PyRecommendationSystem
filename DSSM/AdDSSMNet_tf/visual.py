from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import datetime
import time
import redis
import pickle


# 获取指定时间的时间戳
def get_timestamp(timestr):
    timearray = time.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timearray))
    return timestamp


def handleItem(iterator):
    host = '10.2.0.35'
    # host='10.1.5.107'
    r = redis.StrictRedis(host=host, port=6379, db=0)
    for element in iterator:
        uid_type = element[0]
        p1 = pickle.dump(element[1])
        r.set(uid_type, p1)
        r.expireat(uid_type, expire_timestamp)

    r.close()


def main(*args):
    spark = SparkSession.builder.appName("saveRecomResultToRedis").getOrCreate()

    # 获取推荐结果
    data = spark.sql("select * from algorithm.%s_%s_als_recommedations_list where dt='%s'".format(args[0], args[1], ds))
    data = data.groupBy("uid", "type", "ttype").agg(array(collect_set(col("tuid"))).alias("combineTuid"))
    data = data.groupBy("uid", "type").agg(
        to_json(map_from_arrays(collect_list("ttype"), collect_list("combineTuid"))).alias("recom_list"))
    data = data.withColumn("uidAndType", col("uid") + col("type").cast(StringType))
    data = data.select("uidAndType", "recom_list")

    data.foreachPartition(handleItem)


if __name__ == "__main__":
    # 获取过期的时间
    tomorrow = datetime.datetime.today() + datetime.timedelta(days=1)
    expire_timestamp = get_timestamp(tomorrow.strftime('%Y-%m-%d') + ' 03:00:00')
    main()
