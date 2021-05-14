# -*- coding: utf-8 -*-

import time
import sys

from pyspark import SparkContext
from pyspark.sql import SQLContext

from itertools import combinations
from graphframes import GraphFrame

sc = SparkContext.getOrCreate()

sqlContext = SQLContext(sc)

t = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

start = time.time()

data = sc.textFile(input_file)

header = data.first()

rdd = data.filter(lambda x: x!= header)





def create_candidates(group):
    result = set()
    for pot in group:
        user1 = pot[0]
        user2 = pot[1]

        if user1 > user2:
            user1, user2 = user2, user1

        if (user1, user2) not in result:
            result.add((user1, user2))

    return result

def links(user, business):

    group = list(combinations(user, 2))
    result = create_candidates(group)

    output = []
    for u in result:
        output.append((u, [business]))

    return output


def create_rdd(new_rdd, t):
    return new_rdd.flatMap(lambda x: links(x[1], x[0])).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) >= t)

new_rdd = rdd.map(lambda x: (x.split(",")[1], [x.split(",")[0]])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: list(set(x)))


updated_rdd = create_rdd(new_rdd, t)

def vertices(updated_rdd):
    v = updated_rdd.flatMap(lambda x: [(x[0][0],), (x[0][1],)]).distinct().persist()
    
    return v


def edges(updated_rdd):
    e = updated_rdd.flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])]).persist()
    
    return e


V = vertices(updated_rdd)


E = edges(updated_rdd)


def create_graph(V, E):
    v = sqlContext.createDataFrame(V, ['id'])
    e = sqlContext.createDataFrame(E, ["src", "dst"])
    
    G = GraphFrame(v, e)
    
    return G



community_graph = create_graph(V,E)


"""

def makeListSorted(l):
    l.sort()
    return l




result = (
    community_graph.labelPropagation(maxIter=5)
    .rdd.map(lambda x: (x["label"], [x["id"]]))
    .reduceByKey(lambda a, b: a + b)
    .map(lambda x : makeListSorted(x[1]))
    .sortBy(lambda x : (len(x), x[0]))
    .collect()
)



output = open(output_file, "w")
for community in result:
    community.sort()
    community = ', '.join(["'{}'".format(i) for i in community])
    output.write(community)
    output.write("\n")
output.close()


"""


result = community_graph.labelPropagation(maxIter = 5).collect()
resultDict = {}

for i in result:
    label = i["label"]
    if label not in resultDict.keys():
        resultDict[label] = []
    resultDict[label].append(str(i["id"]))

ordered_keys = sorted(resultDict, key=lambda k: (len(resultDict[k]), min(resultDict[k])))
with open(output_file, "w") as f:
    for key in ordered_keys:
        temp = sorted(resultDict[key])
        f.write("'" + "', '".join(temp) + "'\n")




