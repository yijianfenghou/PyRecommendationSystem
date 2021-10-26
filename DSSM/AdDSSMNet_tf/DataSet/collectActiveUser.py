from pyspark.sql import SparkSession
from pyspark.sql.types import *
import datetime


class CollectActiveUser():

    def __init__(self):
        self.spark = SparkSession.builder.config("spark.sql.warehouse.dir", "/user/hive/warehouse/").appName(
            "collectActiveUser").enableHiveSupport().getOrCreate()
        self.schema = StructType(
            [
                StructField('tuid', StringType()),
                StructField('age', IntegerType()),
                StructField('workid', IntegerType()),
                StructField('height', IntegerType()),
                StructField('sex', StringType()),
                StructField('candidate_set', StringType()),
                StructField('label', IntegerType())
            ]
        )

    def getBeforeDayDate(self, days):
        today = datetime.datetime.now()
        offset = datetime.timedelta(days=-days)
        beforeDate = (today + offset).strftime('%Y-%m-%d')
        return beforeDate + " 00:00:00"

    # / *+ mapjoin(dataplatform_user_action_record) * /
    # 数据库的数据sql
    def getSql(self):
        sql = """
            select
                t1.uid as tuid,
                t1.age as tage,
                t1.work_location as tworkid,
                t1.height as theight,
                t1.sex as tsex
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
                         uid
                     from
                         algorithm.dataplatform_user_action_record t
                     where
                         dt >= "{}"
                         and dt < "{}"
                         and uid is not null
                     group by
                         uid
                )t2
            on cast(t1.uid as string) = t2.uid
            group by
                t1.uid, t1.age, t1.work_location, t1.height, t1.sex
        """.format(self.getBeforeDayDate(60), self.getBeforeDayDate(0))
        return sql

    def get_data(self):
        dataDF = self.spark.sql(self.getSql())

        dataDF = dataDF.toPandas()
        dataDF.to_excel("./item.xlsx", index=False)


if __name__ == "__main__":
    collectData = CollectActiveUser()
    collectData.get_data()
