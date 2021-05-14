
import sys
import numpy as np
import time
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

input_data = sys.argv[1]
test_data = sys.argv[2]
output_file = sys.argv[3]

s = time.time()


data_rdd = sc.textFile(input_data)
test_rdd = sc.textFile(test_data)



header1 = data_rdd.first()
header2 = test_rdd.first()


input_rdd = data_rdd.filter(lambda x: x != header1).map(lambda x: x.split(","))
val_rdd = test_rdd.filter(lambda x: x != header2).map(lambda x: x.split(","))


dict1 = input_rdd.map(lambda x : (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()


dict2 = input_rdd.map(lambda x : (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()


stars_dict = input_rdd.map(lambda x : ((x[0], x[1]), float(x[2]))).collectAsMap()


dict3 = input_rdd.map(lambda x : (x[1], (float(x[2])))).combineByKey(lambda y: (y,1), lambda a,b : (a[0] + b, a[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1] )).mapValues(lambda x: x[0]/x[1]).collectAsMap()



def get_list(X):
    uid, bid = X[0], X[1]

    return uid, bid

def get_corated(uid, userBusiness):
    corated = userBusiness[uid]

    return corated

def compute_pearsons(uid, bid, corated_items, userBusiness, businessUser, ubRating):

    similarity_index = []

    for item in corated_items:

        x = ubRating[(uid, item)]

        corated_user = businessUser[bid].intersection(businessUser[item])

        if (len(corated_user) == 0 or len(corated_user) == 1):
            variance = abs(dict3[bid] - dict3[item])
			
            if (0 <= variance <= 1):
                similarity_index.append([1.0, 1.0*x, 1.0])
                continue

            elif (1 < variance <= 2):
                similarity_index.append([0.5, 0.5*x, 0.5])
                continue				

            else:
                similarity_index.append([0.0, 0.0, 0.0])	
                continue


        temp1 = []
        temp2 = []


        for users in corated_user:


            temp1.append(ubRating[(users, bid)])
            temp2.append(ubRating[(users, item)])

        temp1array = np.asarray(temp1, dtype = np.float32)
        temp2array = np.asarray(temp2, dtype = np.float32)

        b1 = temp1array - dict3[bid]
        b2 = temp2array - dict3[item]


        top = np.sum(np.multiply(b1, b2))
        bottom = np.sqrt(np.sum(b1 ** 2)) * np.sqrt(np.sum(b2 ** 2))

        if (top == 0 or top == 0):
            similarity_index.append([0.0, 0.0, 0.0])	
            continue

        frac = top/bottom

        if (frac < 0):
            continue			

        similarity_index.append([frac, frac * x, abs(frac)])

  
    return similarity_index
  
def check_prediction(pearson, N):

    ordered = sorted(pearson, key = lambda x : -x[0])
    ordered = ordered[ : N]
    matrix = np.array(ordered)
    total = matrix.sum(axis = 0)
 
    return total


def get_pred(chunk, userBusiness, businessUser, ubRating, N):

    uid, bid = get_list(chunk)


    if (bid not in businessUser.keys()):
        return (uid, bid, 3.0)


    if (uid not in userBusiness.keys()):
        return (uid, bid, 3.0)

    corated_items = get_corated(uid, userBusiness)

    pearson = compute_pearsons(uid, bid, corated_items, userBusiness, businessUser, ubRating)

    if (len(pearson) == 0):
        return (uid, bid, 3.0)

    prediction = check_prediction(pearson, N)


    if (prediction[1] == 0.0 or prediction[2] == 0.0):
        return (uid, bid, 3.0)			

    out = prediction[1] / prediction[2]


    return (uid, bid, out)



N = 14

answer = val_rdd.map(lambda chunk : get_pred(chunk, dict1, dict2, stars_dict, N)).collect()



out = 'user_id, business_id, prediction\n'

for item in answer:
    out += str(item[0])+','+str(item[1])+','+str(item[2])+"\n"

with open(output_file, 'w') as f:
    f.write(out)
    


e = time.time() - s

print("Duration: {} sec".format(e))





