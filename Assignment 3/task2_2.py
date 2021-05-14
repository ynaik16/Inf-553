import numpy as np
import sys
from pyspark import SparkContext
import time
import xgboost as xgb

sc = SparkContext.getOrCreate()

general_path = sys.argv[1]

input_data = general_path+'/yelp_train.csv'

test_data = sys.argv[2]

output_file = sys.argv[3]

s = time.time()

train_data = sc.textFile(input_data)

val_data = sc.textFile(test_data)

header_train = train_data.first()

header_val = val_data.first()

train_rdd = train_data.filter(lambda x: x!=header_train).map(lambda x: x.split(','))

test_rdd = val_data.filter(lambda x: x!=header_val).map(lambda x: x.split(','))

user_file = general_path+'/user.json'
userData = sc.textFile(user_file)

business_file = general_path+'/business.json'
BusinessData = sc.textFile(business_file)

userData.take(1)

BusinessData.take(1)

import json



feature1 = userData.map(lambda f: json.loads(f)).map(lambda f : ((f['user_id'], (f['review_count'], f['average_stars'])))).collectAsMap()

feature2 = BusinessData.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()

def get_info(feature, id):
    info = feature[id]

    return info

def extract_details(chunk, feature1, feature2, inp = False):
    
    if (inp):
        uid, bid, rating = chunk[0], chunk[1], -1.0
    else:
        uid, bid, rating = chunk[0], chunk[1], chunk[2]

 #   uid, bid, rating = chunk[0], chunk[1], chunk[2]

    if (uid not in feature1.keys() or bid not in feature2.keys()):
        return [uid, bid, None, None, None, None, None]

    user_review, user_star = get_info(feature1, uid)

    business_review, business_star = get_info(feature2, bid)
	
    return [uid, bid, float(user_review), float(user_star), float(business_review), float(business_star), float(rating)]

yelp_train = train_rdd.map(lambda chunk : extract_details(chunk, feature1, feature2)).collect()

import numpy as np

def convert_to_matrix(x):
    ax = np.array(x)
    return ax

yelp_matrix = convert_to_matrix(yelp_train)



def extract_features(yelp_matrix):

    x = yelp_matrix[:, 2 : -1]
    y = yelp_matrix[:, -1]


    x = np.array(x, dtype = 'float')
    y = np.array(y, dtype = 'float')

    return x, y


train_feature_x, train_feature_y = extract_features(yelp_matrix)


model = xgb.XGBRegressor(objective = 'reg:linear')
model.fit(train_feature_x,train_feature_y)




yelp_test = test_rdd.map(lambda chunk : extract_details(chunk, feature1, feature2, True)).collect()

test_matrix = convert_to_matrix(yelp_test)

test_feature_x, test_feature_y = extract_features(test_matrix)

predictions = model.predict(test_feature_x)

result = np.c_[test_matrix[:, : 2], predictions]


output = 'user_id, business_id, prediction\n'

for item in result:
    output+= str(item[0])+','+ str(item[1])+','+ str(item[2])+'\n'

with open(output_file, 'w') as fo:
    fo.write(output)
    
e = time.time() - s

print("Duration: {} sec".format(e))

