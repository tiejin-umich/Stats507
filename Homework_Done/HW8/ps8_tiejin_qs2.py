import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

user_key = pd.read_csv("hdfs://cavium-thunderx//user/tiejin/user/jbhender/stats507/user_key.csv")
tran = pd.read_csv("stats507/triangles.csv")
rect = pd.read_csv("stats507/rectangles.csv")

spark = SparkSession \
    .builder \
    .appName('my_first_app_name') \
    .getOrCreate()
tran_res = {}
rect_res = {}

user_key = spark.createDataFrame(user_key)
user_key = user_key.filter(user_key.user == "tiejin")
tran = spark.createDataFrame(tran)
rect = spark.createDataFrame(rect)


my_tran = user_key.join(tran,user_key.key==tran.key,'left')
my_rect = user_key.join(rect,user_key.key==rect.key,'left')

my_tran_area =my_tran.withColumn("area",my_tran['base']*my_tran['height']*0.5)
my_rect_area =my_rect.withColumn("area",my_rect['width']*my_rect['length'])

mean_tran = my_tran_area.select(mean('area')).first()[0]
mean_rect = my_rect_area.select(mean('area')).first()[0]

total_tran = my_tran_area.select('area').rdd.map(lambda x:x[0]).reduce(lambda x,y:x+y)
total_rect = my_rect_area.select('area').rdd.map(lambda x:x[0]).reduce(lambda x,y:x+y)

tran_res['number'] = my_tran_area.select('area').rdd.count()
tran_res['total areas'] = total_tran
tran_res['mean areas'] = mean_tran
rect_res['number'] = my_rect_area.select('area').rdd.count()
rect_res['total areas'] = total_rect
rect_res['mean areas'] = mean_rect

final_dict = {"triangles":tran_res,
              "rectangles":rect_res}

final_df = pd.DataFrame(final_dict)
final_df.to_csv("/home/tiejin/ps8_q2_tiejin_results.csv")