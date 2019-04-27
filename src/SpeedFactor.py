import math
import time
import csv

times = []


for _ in range(30):

	time1 = time.time()

	for i in range(0,1000000):
		x = 55.
		x += x
		x /=2
		x *= x
		x = math.sqrt(x)
		x = math.log(x)
		x = math.exp(x)
		x = x/(x+2)

	time2 = time.time()

	elapsed_time = time2-time1

	times.append(elapsed_time)


	print(" * Elapsed time: %f"%(time2-time1))

with open('ASUS_speedtest.csv', 'w', newline='') as file: # Python 3
	w = csv.writer(file, delimiter='\n')        # override for tab delimiter
	w.writerow(times) 