
from pyspark import SparkContext
import random
import sys
import binascii
from blackbox import BlackBox

sc = SparkContext.getOrCreate()
bx = BlackBox()

"""
class BlackBox:

    def ask(self, file, num):
        lines = open(file,'r').readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users

if __name__ == '__main__':
    bx = BlackBox()
    # users = bx.ask()
    # print(users)

"""


input_data = sys.argv[1]

stream_size = int(sys.argv[2])

num_of_asks = int(sys.argv[3])

output_file = sys.argv[4]



def check_prime(val):
	for num in range(2, int(val ** 1/2) + 1):
		if not val % num:
			return False
	return True

def compute(val):
	for num in range(val + 1, val + 10000):
		if check_prime(num): return num


def calculate_hash_values(no_of_hashf, length):

	a = random.sample(range(1, length), no_of_hashf)
	b = random.sample(range(1, length), no_of_hashf)
	
	p = [compute(i) for i in random.sample(range(length, length + 10000), no_of_hashf)]

    
	return {'a' : a, 'b' : b, 'p' : p}

no_of_hashf = 50
params = calculate_hash_values(no_of_hashf, 69997)



def myhashs(user_id):

	hashed_user = []
	user_int = int(binascii.hexlify(user_id.encode('utf8')),16)

	for i in range(no_of_hashf):
		hashed = (((params['a'][i]*user_int + params['b'][i])%params['p'][i])%69997)
		hashed_user.append(hashed)
        
	return hashed_user


groups = 15
rows = int(no_of_hashf / groups)

estimation =0
ground_truth =0

nextptr = 0

def get_hash_fm(input_values):
    hashed_result = []
    user_id = set()

    for id in input_values:

        now = myhashs(id)
        user_id.add(id)

        result = []

        for item in now:
            binary = bin(item)[2:]
            result.append(binary)

        hashed_result.append(result)
    
    return hashed_result, user_id

def compute_zeroes(hashed_result):

    hash_values = []

    for i in range(0, no_of_hashf):
        nulls = -float('inf')
        for j in range(0, len(hashed_result)):
            val = len(hashed_result[j][i]) - len(hashed_result[j][i].rstrip("0"))
            nulls = max(nulls, val)

        hash_values.append(2 ** nulls)

    return hash_values

def compute_estimates(hash_values):
    pred = []
    for i in range(groups):
        total = 0.0
        for j in range(rows):
            total += hash_values[i * rows + j]

        average_pred = total / rows
        pred.append(average_pred)

    pred.sort()

    return pred



def Flajolet_Martin_Algorithm(input_values):

    global groups
    global rows
    global estimation
    global ground_truth
    global nextptr
    hashed_result, user_id = get_hash_fm(input_values)
    hash_values = compute_zeroes(hashed_result)
    pred = compute_estimates(hash_values)


    final_result = pred[int(groups / 2)]

    with open(output_file, "a+") as f:
        f.write(str(nextptr) + ',' + str(len(user_id)) + ',' + str(int(final_result)) + '\n')

    nextptr += 1
    estimation += final_result
    ground_truth += len(user_id)

with open(output_file, "w") as f:
	f.write('Time,Ground Truth,Estimation' + '\n')

for i in range(0, num_of_asks):
	input_vals = bx.ask(input_data, stream_size)
	Flajolet_Martin_Algorithm(input_vals)




