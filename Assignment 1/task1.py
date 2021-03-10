# -*- coding: utf-8 -*-


"""# New Section"""



from pyspark import SparkContext

import json
from operator import add
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]



from pyspark import SparkContext
sc = SparkContext.getOrCreate()

inputrdd = sc.textFile(in_file)

inputrdd.count()

reviews = inputrdd.map(lambda x: json.loads(x))

"""#### Ans 1"""

n_reviews = inputrdd.count()

n_reviews

inputrdd.top(1)

reviews.count()

"""##### Ans 2"""

n_reviews_2018 = reviews.map(lambda x: x['date'][:4]).filter(lambda y: int(y) == 2018).count()

n_reviews_2018

"""#### Ans 3"""

distinct_users = reviews.map(lambda x: x['user_id']).distinct().count()

distinct_users

"""###### ANS 4"""

top_10users= reviews.map(lambda x:(x["user_id"],1)).groupByKey().map(lambda x: [str(x[0]), len(list(x[1]))])\
.sortBy(lambda x: x[1], ascending=False).take(10)

top_10users

#t_10_users = reviews.map(lambda row: (row['user_id'], row['review_id'])).groupByKey().map(lambda x: [str(x[0]), len(list(x[1]))])\
#.sortBy(lambda x: x[1], ascending=False).take(10)

#t_10_users



"""#### Ans 5"""

distinct_business = reviews.map(lambda x: x['business_id']).distinct().count()
distinct_business

"""#### Ans 6"""

top_10business= reviews.map(lambda x:(x["business_id"],1)).groupByKey().map(lambda x: [str(x[0]), len(list(x[1]))])\
.sortBy(lambda x: x[1], ascending=False).take(10)

top_10business

newdict = {"n_review":n_reviews,
          "n_review_2018":n_reviews_2018,
          "n_user":distinct_users,
          "top10_user":top_10users,
          "n_business":distinct_business,
          "top10_business":top_10business}

with open(out_file, 'w+') as f:
    json.dump(newdict, f)

