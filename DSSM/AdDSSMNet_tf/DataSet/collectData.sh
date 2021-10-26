export HADOOP_USER_NAME=work
export PYSPARK_PYTHON=/usr/local/python3/bin/python3.7
export PYSPARK_DRIVER_PYTHON=/usr/local/python3/bin/python3.7

ds=`date "+%Y%m%d"`

/opt/cloudera/parcels/CDH/bin/spark-submit --driver-memory 4G --executor-memory 4G  --conf spark.cores.max=8 /home/work/algorithm/JyRecommendationSystem/TwoTowerModel/DataProgress/collectHiveData.py

/opt/cloudera/parcels/CDH/bin/spark-submit --driver-memory 8G --executor-memory 4G  --conf spark.cores.max=8 /home/work/algorithm/JyRecommendationSystem/TwoTowerModel/DataProgress/collectActiveUser.py

echo "sucess"