import random
import sys
from pyspark import SparkContext
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


def set_parameters():
    global filter_bit_Array
    global traversed
    global false_positives
    global true_negatives
    global next

    return filter_bit_Array, traversed, false_positives, true_negatives, nextptr

def calculate_fpr(false_positives, true_negatives):

    return (false_positives / (false_positives + true_negatives))

filter_bit_Array = [0] * 69997
traversed =set()
false_positives = 0.0
true_negatives = 0.0
nextptr = 0

def BloomFiltering(user_id):

    global filter_bit_Array
    global traversed
    global false_positives
    global true_negatives
    global nextptr


    for u in user_id:

        now = myhashs(u)
        ctr = 0
        for i in range(no_of_hashf):
            if (filter_bit_Array[now[i]] == 1):
                ctr += 1

        if u not in traversed:
            if (ctr == no_of_hashf):
                false_positives += 1.0
            else:
                true_negatives += 1.0	

        for i in range(no_of_hashf):
            filter_bit_Array[now[i]] = 1

        traversed.add(u)

    FPR = calculate_fpr(false_positives, true_negatives)


    with open(output_file, "a+") as out:
        out.write(str(nextptr) + ',' + str(FPR) + '\n')

    nextptr += 1

def run_algorithm(num_of_asks, input_data, stream_size, bx):

    for i in range(0, num_of_asks):
        input_par = bx.ask(input_data, stream_size)
        BloomFiltering(input_par)

    return

with open(output_file, "w") as f:
	f.write('Time,FPR' + '\n')
 

for i in range(0, num_of_asks):
	item = bx.ask(input_data, stream_size)
	BloomFiltering(item)

