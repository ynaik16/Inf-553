

from pyspark import SparkContext
from collections import deque
import time
import sys
from itertools import combinations
from copy import deepcopy

sc = SparkContext.getOrCreate()

t = int(sys.argv[1])
input_file = sys.argv[2]
betweeness_calc = sys.argv[3]
community_file = sys.argv[4]

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

start = time.time()

rdd = sc.textFile(input_file)
header = rdd.first()

new_rdd = rdd.filter(lambda x: x!= header).map(lambda x: (x.split(",")[1], [x.split(",")[0]])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: list(set(x))).flatMap(lambda x: links(x[1], x[0])).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) >= t)

vertices = new_rdd.flatMap(lambda x: [(x[0][0],), (x[0][1],)]).distinct()
unidercted_edge = new_rdd.map(lambda x: (x[0][0], x[0][1]))
biderected_edge = new_rdd.flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])])

class CustomGraphFrame:

    adjacents = {}
    v = {}
    e = {}
    pair = {}

    # variables for computing
    traversed = []
    parent_node = {}
    dist = {}
    between = {}

    # Modularity
    Q = float("-inf")
    combs = []

    def __init__(self, pair, v, e):
        self.pair = pair
        self.v = v
        self.e = e

    def betweeness(self):
        for first in self.v:
            self.reset()
            self.BREDTH_FIRST_SEARCH(first)
            self.reverse()
            curr_bns = self.getcurr_bns()
            for edge in curr_bns:
                self.e[edge] = self.e[edge] + curr_bns[edge]
        for edge in self.e:
            self.e[edge] = self.e[edge] / 2
        self.reset()
        self.between = self.e

    def Modularity(self, group):

        QGS = 0
        m = len(self.e)

        for nbr in group:
            if len(nbr) == 1:
                continue
            for i in nbr:
                for j in nbr:
                    if i == j:
                        continue
                    A_ij = 0
                    if j in self.adjacents[i] and i in self.adjacents[j]:
                        A_ij = 1
                    k_i = len(self.adjacents[i])
                    k_j = len(self.adjacents[j])
                    QGS = QGS + (A_ij - (k_i * k_j) / (2 * m))
        QGS = QGS / (2 * m)
        return QGS
    
    def check_for_edge(self):
        adj = self.pair
        for vertex in adj:
            if len(adj[vertex]) > 0:
                return True
        return False

################################################################################


    def remove_pair(self, pairs):
        for p in pairs:
            x = p[0]
            y = p[1]
            if y in self.pair[x]:
                self.pair[x].remove(y)
            if x in self.pair[y]:
                self.pair[y].remove(x)
        
    def betweeness_score(self):

        max_bns = []
        bns = self.return_betweeness()
        score = max(bns, key=lambda x: x[2])[2]
        for item in bns:
            if abs(item[2] - score) <= 0.00000001:
                max_bns.append((item[0], item[1]))
                break
    
        return max_bns


    def find(self):
        self.adjacents = deepcopy(self.pair)  # store
        while(self.check_for_edge()):
            self.create_neighbors()
            bns_stored = self.betweeness_score()
            self.remove_pair(bns_stored)
        return self.combs

    def create_neighbors(self):
        val = []
        vertex = set(self.v)  # all vertices
        while len(vertex) > 0:
            traversed, super_node, distance = self.BREDTH_FIRST_SEARCH(list(vertex)[0])
            traversed = list(traversed)
            traversed.sort()
            val.append(list(traversed))
            vertex = vertex - set(traversed)
        
        q = self.Modularity(val)
        if q > self.Q:
            self.Q = q
            self.combs = val

    def reset(self):
        self.vistit_order = []
        self.parent_node = {}
        self.dist = {}
        self.between = {}

    def getcurr_bns(self):
        return self.between

    def return_edges(self):
        return self.e

    def BREDTH_FIRST_SEARCH(self, root, save=True):
        traversed=deque([])
        parent_node = {}
        dist= {}
        queue = deque([root])
        curr_level = -1
        parent_node[root] = []
        dist[root] = {"dist": 0, "ctr": 1}

        while queue:
            curr_level = curr_level + 1
            size = len(queue)
            for i in range(size):
                now = queue.popleft()
                traversed.append(now)

                adj_node = self.pair[now]
                for node in adj_node:
                    # Calculate the short distance and count
                    if node not in dist:
                        dist[node] = {
                            "dist": curr_level + 1,
                            "ctr": dist[now]["ctr"],
                        }
                    elif (curr_level + 1) == dist[node]["dist"]:
                        dist[node]["ctr"] = (
                            dist[node]["ctr"]
                            + dist[now]["ctr"]
                        )

                    if node not in parent_node:
                        # not visted:
                        queue.append(node)
                        parent_node[node] = []
                    if node not in parent_node[now] and curr_level < dist[node]["dist"]:
                        parent_node[node].append(now)

        self.traversed = traversed
        self.parent_node = parent_node
        self.dist = dist
        return traversed, parent_node, dist

    def reverse(self):

        between= {}
        sequence=self.traversed
        parent_node=self.parent_node
        dist = self.dist
        x = {}
        for vx in sequence:
            x[vx] = 1

        for vx in list(reversed(sequence)):
            for node in parent_node[vx]:
                ewt = x[vx] * (dist[node]["ctr"] / dist[vx]["ctr"])
                v1 = vx
                v2 = node
                if v1 > v2:
                    v1, v2 = v2, v1
                if (v1, v2) not in between:
                    between[(v1, v2)] = 0
                between[(v1, v2)] = between[(v1, v2)] + ewt
                x[node] = x[node] + ewt

        self.between = between
    
    def return_betweeness(self):
        for curr_edge in self.e:
            self.e[curr_edge] = 0
        self.betweeness()
        edges = self.return_edges()
        bns = []
        for i in edges:
            bns.append((i[0], i[1], edges[i]))
        bns = sorted(bns, key=lambda x: (-x[2], x[0], x[1]))
        return bns

vs = set(vertices.map(lambda x: x[0]).collect())
edges = unidercted_edge.map(lambda x: (x, 0)).collectAsMap()
adjacency_list = biderected_edge.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).collectAsMap()

graph = CustomGraphFrame(adjacency_list, vs, edges)





def get_betweeness(graph):
    return graph.return_betweeness()

bns_val = get_betweeness(graph)

result1 = open(betweeness_calc, "w")
for i in bns_val:
    result1.write("('" + i[0] + "', '" + i[1] + "'), " + str(i[2])+"\n")
result1.close()

def community_detection(graph):
    community = list(graph.find())

    return community



result_community = community_detection(graph)
    # best_partition = sorted(best_partition, key=lambda x: (len(x),list(x)[0]))

out = []
for c in result_community:
    val = list(c)
    out.append(val)

updated_community = out
sorted_community = sorted(updated_community, key=lambda x: (len(x), x[0]))

result2 = open(community_file, "w")
for user in sorted_community:
    user = list(user)
    user.sort()
    output = ", ".join(["'{}'".format(i) for i in user])
    result2.write(output+ "\n")
result2.close()

end = time.time() -start

print("Duration: {} sec".format(end)) 