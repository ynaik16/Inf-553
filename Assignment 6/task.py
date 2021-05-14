import numpy as np
from sklearn.cluster import KMeans
import time
import os
import collections
import math
import sys
import ast



def compute_statistics(allpoints):
    try:
        N = len(allpoints)
        SUM = [sum(x) for x in zip(*allpoints)]
        tota = []
        for x in zip(*allpoints):
            SUMSQ = 0
            for i in x: 
                SUMSQ = SUMSQ + i*i
            tota.append(SUMSQ)
    except:
        N =1
        SUM = list(allpoints)
        tota = []
        for x in range(len(SUM)):
            SUMSQ = SUM[x] * SUM[x]
            tota.append(SUMSQ)
    return [N,SUM,tota]


    
input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]
    
start = time.time()
#n_cluster = 10
percentage = 0.2
inputData = open(input_file,"r")

def read(read_data):

    value = [ x.strip() for x in read_data.readlines()]

    line = []
    for v in value:
        temp = []
        temp = v.split(",")
        string = []
        for t in temp:
            string.append(float(t))
        line.append(string)
    
    return line

data = read(inputData)


#Step 1: load 20% of data
Expected_result = {}
array_d = np.array(data)
#y = npData[:,1:2]

total_points = len(array_d)

threshold = 0.2

starting_data = array_d[:int(total_points*threshold)]
sample = {}

for x in starting_data:
    sample[repr(list(x[2:]))] = int(x[0])

initial_list = list(sample.keys())

#Step 2: kmeans on all data

kmeans_stored = []
for i in initial_list:
    kmeans_stored.append(ast.literal_eval(i))

matrix1 = np.array(kmeans_stored)
   
KMEANS_one = KMeans(n_clusters=10*n_cluster, random_state=0).fit(kmeans_stored)
LABELS_one = dict(collections.Counter(KMEANS_one.labels_))

#step 3: assigning clusters with 1 point to RS
Retained_group = []

discard_list2 = []
for x in LABELS_one:
    
    if LABELS_one[x] == 1 :
        
        index = np.where(KMEANS_one.labels_==x)
        points = matrix1[index][0]
        Retained_group.append(points)
        
    else:
        index = np.where(KMEANS_one.labels_==x)
        for i in index[0]:
            #print(i)
            points = matrix1[i]
            discard_list2.append(points)
        
matrix2 = np.array(discard_list2)
   
#step 4: kmeans on the remaining data
KMEANS_two = KMeans(n_clusters=n_cluster, random_state=0).fit(matrix2)
LABELS_two = dict(collections.Counter(KMEANS_two.labels_))

discard_id = {}
discard_group = {}

for x in LABELS_two:
    index = np.where(KMEANS_two.labels_==x)
    tempIndex = []
    tempCoords = []
    for i in index[0]:
        #print(i)
        coords = matrix2[i]
        dictIndex = int(sample.get(repr(list(coords))))
        tempIndex.append(dictIndex)
        tempCoords.append(coords)
    discard_id[x] = tempIndex
    discard_group[x] = compute_statistics(tempCoords)

#Step 6: Running kmeans on RS and getting final RS and CS
KMEANS_three = KMeans(n_clusters=n_cluster, random_state=0).fit(Retained_group)
LABELS_three = collections.Counter(KMEANS_three.labels_)

updated_Retained = []
compressed_id = {}
compressed_group = {}

for x in LABELS_three:
    if LABELS_three[x]==1:
        index = np.where(KMEANS_three.labels_== x)
        #print('1',index)
        coords = Retained_group[index[0][0]]
        updated_Retained.append(coords)
    else:
        index = np.where(KMEANS_three.labels_==x)
        tempIndex = []
        tempCoords = []
        for i in index[0]:
            #print(i)
            coords = Retained_group[i]
            dictIndex = int(sample.get(repr(list(coords))))
            tempIndex.append(dictIndex)
            tempCoords.append(coords)
        compressed_id[x] = tempIndex
        compressed_group[x] = compute_statistics(tempCoords)
        
Retained_group = updated_Retained

updated_Retained = []

RS = {}

for x in Retained_group:
    dictIndex = int(sample.get(repr(list(x))))
    RS[repr(list(x))] = int(dictIndex)
    
loopctr = 1
n_discard = 0
n_cluster_compressed = 0
n_compressed = 0
n_retained = 0
for k,v in discard_group.items():
     n_discard += v[0]
for k,v in compressed_group.items():
     n_compressed += v[0]
        
n_cluster_compressed = len(compressed_group)
n_retained = len(RS)

Expected_result[loopctr] = [n_discard,n_cluster_compressed,n_compressed,n_retained]

#Step 7: Load another 20% of the data

begin = int(total_points * threshold)
finish = begin + int(total_points * threshold)
d = 10



def standard_deviation(statistic):
    
    dev = {}
    
    for k,v in statistic.items():
        N = v[0]
        SUM = v[1]
        SUMSQ = v[2]
        tot = []
        for i in range(len(SUM)):
            mean = SUM[i]/N
            x = mean * mean
            y = SUMSQ[i]/N
            diff = y-x
            std = math.sqrt(diff)
            tot.append(std)
        dev[k] = tot
        
    return dev


def Mahalanobis_Distance(coordinate, centroid, deviation):
    #MD = 0
    temp = 0
    for x in range(len(centroid)):
        numerator = coordinate[x] - centroid[x]
        denominator = deviation[x]
        frac = numerator/denominator
        sq = frac*frac
        temp = temp + sq
    answer = math.sqrt(temp)
    return answer


def find_centroid(statistics):
    try:
        centroids = {}
        for k,v in statistics.items():
            length = v[0]
            individ = []
            for i in range(len(v[1])):
                avg = v[1][i]/length
                individ.append(avg)
            centroids[k] = individ
        return centroids
    
    except:
        N = statistics[0]
        centroid = []
        for x in range(len(statistics[1])):
            avg = statistics[1][x]/N
            centroid.append(avg)
        return centroid

def change(group,coordinate):
    N = group[0] + coordinate[0]
    total = []
    Square = []
    SUM = 0
    SUMSQ = 0
    for x in range(len(group[1])):
        SUM = group[1][x] + coordinate[1][x]
        SUMSQ = group[2][x] + coordinate[2][x]
        total.append(SUM)
        Square.append(SUMSQ)
    return [N,total,Square]
        


while begin < total_points:
    
    starting_data = array_d[begin:finish]
    sample = {}
    
    for x in starting_data:
        sample[repr(list(x[2:]))] = int(x[0])

    temp_sample = list(sample.keys())
    temp_final = []
    for i in temp_sample:
        temp_final.append(ast.literal_eval(i))

    temp_final = np.array(temp_final)
    
    ctd_discard = find_centroid(discard_group)
    ctd_compressed = find_centroid(compressed_group)
    dev_discard = standard_deviation(discard_group)
    dev_compressed = standard_deviation(compressed_group)

    for x in temp_final:
        #Step 8: for each incoming point compare the Mahalnobis distance between DS centroids and the point
        mini = 2*(math.sqrt(d))+1
        mergeCluster = 0

        
        for dc,v in ctd_discard.items():
            mahalanobisDistance = Mahalanobis_Distance(list(x),v,dev_discard[dc])
            if mahalanobisDistance < mini:
                mini = mahalanobisDistance
                mergeClusterDS = dc
                
        if mini < 2*(math.sqrt(d)):
            
            xSummary = compute_statistics(x)
            dictIndex = int(sample.get(repr(list(x))))
            discard_id[mergeClusterDS].append(dictIndex)
            discard_group[mergeClusterDS] = change(discard_group[mergeClusterDS],xSummary)
          
            
        else:
            #Step 9: for each incoming point compare the mahalanobis distance between CS centroids and the point
            mini = 2*(math.sqrt(d))+1
            
            for cc,v in ctd_compressed.items():
                mahalanobisDistanceCS = Mahalanobis_Distance(list(x),v,dev_compressed[cc])
                
                if mahalanobisDistanceCS < mini:
                    mini = mahalanobisDistanceCS
                    mergeClusterCS = cc
                    
            if mini < 2*(math.sqrt(d)):
                
                dictIndex = int(sample.get(repr(list(x))))
                compressed_id[mergeClusterCS].append(dictIndex)
                xSummary = compute_statistics(x)
                compressed_group[mergeClusterCS] = change(compressed_group[mergeClusterCS],xSummary)
            else:
                #Step 10: Assign point to RS
                dictIndex = int(sample.get(repr(list(x))))
                RS[repr(list(x))] = int(dictIndex)
                Retained_group.append(x)

    #Step 11: Run Kmeans on remaining RS and separate RS and CS Clusters
    if len(RS) > n_cluster:
        KMEANS_four = KMeans(n_clusters=n_cluster, random_state=0).fit(Retained_group)
        LABELS_four = collections.Counter(KMEANS_four.labels_)

        updated_Retained = []
        
        for x in LABELS_four:
            if LABELS_four[x]==1:
                index = np.where(KMEANS_four.labels_== x)
                #print('1',index)
                coords = Retained_group[index[0][0]]
                updated_Retained.append(coords)
                
            else:
                index = np.where(KMEANS_four.labels_==x)
                tempIndex = []
                tempCoords = []
                
                for i in index[0]:
                    #print(i)
                    coords = Retained_group[i]
                    dictIndex = int(RS.get(repr(list(coords))))
                    tempIndex.append(dictIndex)
                    tempCoords.append(coords)
                    
                mini = 2*(math.sqrt(d))+1
                mergeClusterNew = 0
                
                for cc,v in ctd_compressed.items():
                    
                    newCSCluster = compute_statistics(tempCoords)
                    newCSCentroid = find_centroid(newCSCluster)
                    CSDistance = Mahalanobis_Distance(newCSCentroid, v, dev_compressed[cc])
                    
                    if CSDistance < mini:
                        mini = CSDistance
                        mergeClusterNew = cc
                        
                if mini < 2*(math.sqrt(d)):
                    #Step 12: Merging CS Clusters that have a Mahalanobis Distance of < 2*sqrt(d)
                   
                    xSummary = newCSCluster
                    compressed_group[mergeClusterNew] = change(compressed_group[mergeClusterNew],xSummary)
                    
                    if len(tempIndex)>1:
                        for x in tempIndex:
                            compressed_id[mergeClusterNew].append(x)
                            
                else:
                    
                    key = max(compressed_group.keys()) + 1
                    compressed_id[key] = tempIndex
                    compressed_group[key] = compute_statistics(tempCoords)

        Retained_group = updated_Retained
        updated_Retained = []
        
        dict8 = {}
        
        for x in Retained_group:
            try:
                dictIndex = int(RS.get(repr(list(x))))
                dict8[repr(list(x))] = int(dictIndex)
            except:
                dictIndex = int(sample.get(repr(list(x))))
                dict8[repr(list(x))] = int(dictIndex)

        RS = dict8
        dict8 = {}
        
    begin = finish
    finish = begin + int(total_points * threshold)
    
    if finish > total_points:
        
        #Last Step: Merge CS Clusters with DS Clusters
        
        ctd_discard = find_centroid(discard_group)
        ctd_compressed = find_centroid(compressed_group)
        dev_discard = standard_deviation(discard_group)
        dev_compressed = standard_deviation(compressed_group)
        
        
        EndResult = {}

        for k,v in ctd_compressed.items():
            mini = 2*(math.sqrt(d))+1
            mergeCluster = 0
            
            
            for k1,v1 in ctd_discard.items():
                maDistance = Mahalanobis_Distance(v,v1,dev_discard[k1])
                if maDistance < mini:
                    mini = maDistance
                    mergeCSDS = k1
                    
            if mini < 2*(math.sqrt(d)):
                CSIndexes = CSClusterIndex[k]
                if len(CSIndexes)>1:
                    for x in CSIndexes:
                        discard_id[mergeClusterDS].append(x)
                discard_group[mergeClusterDS] = change(discard_group[mergeClusterDS],xSummary)
                #DSAppendDict[mergeClusterDS].append(x)
            else:
                EndResult[k] = compressed_group[k]
        
        loopctr += 1
        n_discard = 0
        n_cluster_compressed = 0
        n_compressed = 0
        n_retained = 0
        
        for k,v in discard_group.items():
             n_discard += v[0]
        for k,v in EndResult.items():
             n_compressed += v[0]

        n_cluster_compressed = len(EndResult)
        n_retained = len(RS)

        Expected_result[loopctr] = [n_discard,n_cluster_compressed,n_compressed,n_retained]


    else:
        
        loopctr += 1
        n_discard = 0
        n_cluster_compressed = 0
        n_compressed = 0
        n_retained = 0
        for k,v in discard_group.items():
             n_discard += v[0]
        for k,v in compressed_group.items():
             n_compressed += v[0]

        n_cluster_compressed = len(compressed_group)
        n_retained = len(RS)

        Expected_result[loopctr] = [n_discard,n_cluster_compressed,n_compressed,n_retained]
        
        
output_res = {}

for k,v in discard_id.items():
    for i in v:
        output_res[i] = k
for k,v in compressed_id.items():
    for i in v:
        output_res[i] = -1
for k,v in RS.items():
    output_res[v] = -1
    
res = dict(sorted(output_res.items()))

f = open(output_file, "w")
f.write("The intermediate results:\n")
for k,v in Expected_result.items():
    f.write("Round " + str(k) + ": " + str(v[0]) + "," + str(v[1]) + "," + str(v[2]) + "," + str(v[3]) + "\n")
f.write("The clustering results:\n")
for k,v in res.items():
    f.write( str(k)+ "," + str(v) + "\n" )
f.close()


e = time.time() - start

print("Duration : ",e)
