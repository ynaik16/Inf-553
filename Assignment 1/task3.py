# -*- coding: utf-8 -*-
"""task3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tbxg83Wjic37EFDRVjxkOPb8raJSwR41
"""

#!apt-get install openjdk-8-jdk

#!wget -q https://archive.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz

#!tar xf spark-2.4.4-bin-hadoop2.7.tgz

#!pip install -q findspark

#import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
#os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import json
from operator import add
import findspark
findspark.init()
import sys

#from pyspark.sql import SparkSession

#spark = SparkSession.builder\
#        .master("local")\
#        .appName("Colab")\
#        .config('spark.ui.port', '4050')\
#        .getOrCreate()

#spark


from pyspark import SparkContext



#from google.colab import drive
#drive.mount('/content/drive')

sc = SparkContext.getOrCreate()



review_input = argv[1]
business_input = argv[2]

output_file_path_a = argv[3]
output_file_path_b = argv[4]

"""##### part a"""


rvs = spark             \
        .sparkContext           \
        .textFile(review_input) \
        .coalesce(27)            \
        .map(json.loads)        \
        .map(lambda x: (x["business_id"], x["stars"]))

bns = spark              \
        .sparkContext               \
        .textFile(business_input)   \
        .coalesce(27)                \
        .map(json.loads).map(lambda x: (x["business_id"], x["city"]))

combined = rvs                               \
        .join(bns)                               \
        .coalesce(2)                                    \
        .map(lambda x: [x[1][1], x[1][0]])              \
        .groupByKey()                                   \
        .map(lambda x: [x[0], (sum(x[1])/len(x[1]))])           \
        .sortBy(lambda x: (-x[1], x[0]))

with open(output_file_path_a,"w+") as of:
    of.write("city, stars \n")
    for data in combined.collect():
        of.write(""+str(data[0])+","+str(data[1])+"\n")

"""## part b

"""

import time

answer = {}

#   Python implementation

start_time = time.time()
m1 = combined.collect()

for data in m1[:10]:
    print(data)

exec_time = time.time() - start_time
answer["m1"] = exec_time



# Spark implementation

start_time = time.time()
m2 = combined.take(10)
for data in m2:
    print(data)

exec2_time = time.time() - start_time
answer["m2"] = exec2_time


# Printing the cities

#with open("city_ratings", "w") as f:
#    f.write("city,stars\n")
#    for x in m1:
#        f.write(f"{x[0]},{x[1]}\n")


# Printing operations result

with open(output_file_path_b, "w") as of:
    json.dump(answer, of)

