import sys
import random
import binascii

class BlackBox:

    def ask(self, file, num):
        lines = open(file,'r').readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users

if __name__ == '__main__':
    bx = BlackBox()
    random.seed(553)



#bx = BlackBox()

input_data = sys.argv[1]

stream_size = int(sys.argv[2])

num_of_asks = int(sys.argv[3])

output_file = sys.argv[4]

#random.seed(553)

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





result = []

items = 0

def Fixed_Size_Sampling(input_values):
    
    #random.seed(553)

    global items
    global result

    for idval in input_values:
        
        items += 1

        if (len(result) < 100):
            result.append(idval)
           
       

        else:	
            index = random.random()
            
            check = float(100/items)
            
            #print("probability ={} , s/n = {}".format(index,check))

            if (index < check):
        
                nth_user = random.randint(0,100000) % 100
            
                result[nth_user] = idval

    if (items != 0 and items % 100 == 0):

        with open(output_file, "a+") as out:
#            print(items, result[0], result[20], result[40], result[60], result[80])
            
            out.write(str(items) + ',' + str(result[0]) + ',' + str(result[20]) + ',' + str(result[40]) + ',' + str(result[60]) + ',' + str(result[80]) + '\n')

with open(output_file, "w") as out:
	out.write('seqnum,0_id,20_id,40_id,60_id,80_id' + '\n')

for i in range(0, num_of_asks):
	inp = bx.ask(input_data, stream_size)
	Fixed_Size_Sampling(inp)
