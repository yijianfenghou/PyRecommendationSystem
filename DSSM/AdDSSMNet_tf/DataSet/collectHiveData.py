from pyspark.sql import SparkSession
import datetime
import pandas as pd
from pyspark.sql.types import *


class CollectHiveData():

    def __init__(self):
        self.spark = SparkSession.builder.config("spark.sql.warehouse.dir", "/user/hive/warehouse/").appName(
            "collectUidAndTUidData").enableHiveSupport().getOrCreate()
        self.schema = StructType(
            [
                StructField('uid', StringType()),
                StructField('age', IntegerType()),
                StructField('workid', IntegerType()),
                StructField('height', IntegerType()),
                StructField('sex', StringType()),
                StructField('candidate_set', StringType()),
                StructField('label', IntegerType())
            ]
        )

        # self.schema = StructType(
        #     [
        #         StructField('uid', StringType()),
        #         StructField('age', IntegerType()),
        #         StructField('work_location', IntegerType()),
        #         StructField('height', IntegerType()),
        #         StructField('sex', StringType()),
        #         StructField('subeventid', StringType()),
        #         StructField('opposite_age', IntegerType()),
        #         StructField('opposite_work_location', IntegerType()),
        #         StructField('opposite_height', IntegerType()),
        #         StructField('opposite_sex', StringType()),
        #         StructField('tuidEmbed', ArrayType(StringType(), containsNull=True)),
        #         StructField('label', IntegerType())
        #     ]
        # )

    def getBeforeDayDate(self, days):
        today = datetime.datetime.now()
        offset = datetime.timedelta(days=-days)
        beforeDate = (today + offset).strftime('%Y-%m-%d')
        return beforeDate

    # / *+ mapjoin(dataplatform_user_action_record) * /
    # 数据库的数据sql
    def getSql(self):
        # sql = '''
        #     select
        #         t1.uid,
        #         t1.age,
        #         t1.work_location,
        #         t1.height,
        #         t1.sex,
        #         t3.subeventid,
        #         t3.opposite_age,
        #         t3.opposite_work_location,
        #         t3.opposite_height,
        #         t3.opposite_sex,
        #         t2.timestamp,
        #         if(t2.eventid == "8.32", 0, 1) as label
        #     from
        #         (
        #             select
        #                 uid,
        #                 cast(year(CURRENT_DATE) as int) - cast(birth_year as int) as age,
        #                 work_location,
        #                 height,
        #                 sex
        #             from
        #                 algorithm.users_search
        #             where
        #                 gid >= 3
        #                 and gid <= 10
        #                 and uid is not null
        #                 and work_location = '51'
        #          )t1
        #     inner join
        #         (
        #              select
        #                  uid,
        #                  eventid,
        #                  subeventid,
        #                  unix_timestamp(`time`) as timestamp
        #              from
        #                 algorithm.dataplatform_user_action_record
        #              where
        #                 dt >= %s
        #                 and dt < %s
        #                 and eventid in ("8.32", "8.33", "8.34")
        #                 and subeventid is not null
        #         )t2
        #         on cast(t1.uid as string) = t2.uid
        #     left join
        #         (
        #             select
        #                 uid as subeventid,
        #                 cast(year(CURRENT_DATE) as int) - cast(birth_year as int) as opposite_age,
        #                 work_location as opposite_work_location,
        #                 height as opposite_height,
        #                 sex as opposite_sex
        #             from
        #                 algorithm.users_search
        #             where
        #                 gid >= 3
        #                 and gid <= 10
        #                 and uid is not null
        #             group by
        #                 uid,cast(year(CURRENT_DATE) as int)-cast(birth_year as int),work_location,height,sex
        #         )t3
        #         on cast(t2.subeventid as string) = cast(t3.subeventid as string)
        #     group by
        #         uid, age, work_location, height, sex
        #     order by
        #         timestamp
        # '''.format(self.getBeforeDayDate(), "2021-03-28")
        sql = """
            select
                t1.uid as uid,
                t1.age as age,
                t1.work_location as workid,
                t1.height as height,
                t1.sex as sex,
                concat_ws("&",collect_list(subeventid)) as candidate_set,
                if(t2.eventid == "8.32", 0, 1) as label
            from
                (
                    select
                        uid,
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
                         t.`time` 
                     from
                         algorithm.dataplatform_user_action_record t
                     where
                         dt >= "{}"
                         and dt < "{}"
                         and eventid in ("8.32", "8.33", "8.34")
                         and subeventid is not null
                         and length(subeventid) > 0 
                )t2
            on cast(t1.uid as string) = t2.uid
            group by
                t1.uid, t1.age, t1.work_location, t1.height, t1.sex, t2.eventid
        """.format(self.getBeforeDayDate(15), self.getBeforeDayDate(0))
        return sql

    def get_data(self):
        dataDF = self.spark.sql(self.getSql())

        dataDF = dataDF.toPandas()
        # df = self.spark.createDataFrame(dataRdd, schema= self.schema)
        # dataDF = df.toPandas()
        # dataDF.dropna(subset=['subeventid'], inplace=True)
        # dataDF['subeventid'] = dataDF['subeventid'].astype('int64')
        # dataDF['uid'] = dataDF['uid'].astype('int64')

        # tuidDF = pd.read_excel("./aaaa.xlsx", columns=['subeventid', 'pid', 'tuidEmbed'])
        # tuidDF['pid'] = tuidDF['pid'].astype('int64')
        # tuidDF['subeventid'] = tuidDF['subeventid'].astype('int64')
        # result = pd.merge(dataDF, tuidDF, on=['subeventid'], how='inner')
        dataDF.to_excel("./user.xlsx", index=False)


if __name__ == "__main__":
    collectData = CollectHiveData()
    collectData.get_data()
