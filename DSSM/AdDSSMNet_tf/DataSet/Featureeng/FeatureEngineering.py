from pyspark.ml.feature import StringIndexer, QuantileDiscretizer
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType


def labelEncoderExample(df, df1, feature):
    sel_uid_feature_col = df.select(feature)
    if feature == "uid":
        sel_subeventid_feature_col = df.select("subeventid").withColumnRenamed("subeventid", feature)
        sel_uid_feature_col = sel_uid_feature_col.unionAll(sel_subeventid_feature_col)
    sel_tuid_feature_col = df1.select("t"+feature).withColumnRenamed("t"+feature, feature).select(feature)
    combine_feature = sel_uid_feature_col.unionAll(sel_tuid_feature_col).dropDuplicates([feature])
    indexer = StringIndexer(inputCol=feature, outputCol=feature+"Index")
    indexed = indexer.fit(combine_feature).transform(combine_feature)
    indexed = indexed.withColumn(feature+"Index", indexed[feature+"Index"].cast(IntegerType())).dropDuplicates([feature])
    df = indexed.join(df, df[feature] == indexed[feature], how="inner").drop(indexed[feature])
    if feature == "uid":
        df = df.withColumnRenamed(feature + "Index", "mid" + feature + "Index")
        df = indexed.join(df, df["subeventid"] == indexed[feature], how="inner").drop(indexed[feature])
        df = df.withColumnRenamed(feature + "Index", "subeventid" + "Index").withColumnRenamed("mid" + feature + "Index", feature + "Index")
    df1 = indexed.join(df1, df1["t"+feature].cast(StringType()) == indexed[feature].cast(StringType()), how="inner").drop(indexed[feature])
    df1 = df1.withColumnRenamed(feature + "Index", "t" + feature + "Index")
    return df, df1


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
    df1 = df1.withColumnRenamed("heightIndex", "theight")
    df1 = df1.withColumnRenamed("sexIndex", "tsex")
    return df, df1



