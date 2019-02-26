import numpy as np
import matplotlib.pyplot as plt
import math
data = np.genfromtxt('kmeans_data.txt')

x_val = data[:,0]
y_val = data[:,1]
plt.scatter(x_val,y_val, marker='o', c = 'blue')
plt.show()

x_mean1 = data[0,0]
y_mean1 = data[0,1]
x_mean2 = data[1,0]
y_mean2 = data[1,1]

trans_mean1 = np.sqrt(x_mean1 ** 2 + y_mean1 ** 2)
trans_mean2 = np.sqrt(x_mean2 ** 2 + y_mean2 ** 2)

trans_data = np.zeros(len(x_val))
for i in range(len(x_val)):
	trans_data[i] = x_val[i] ** 2 + y_val[i] ** 2

cluster_id = np.zeros(len(x_val))

prev_trans_mean1 = 0
prev_trans_mean2 = 0
count = 0
while count<=50:
	for i in range(len(x_val)):
		pt = trans_data[i]
		dist1 = np.linalg.norm(pt - trans_mean1)
		dist2 = np.linalg.norm(pt - trans_mean2)
		if(dist1 < dist2):
			cluster_id[i] = 1
		else:
			cluster_id[i] = 2

	sum1=0
	sum2=0
	count1=0
	count2=0
	for i in range(len(trans_data)):
		if cluster_id[i] == 1:
			sum1 += trans_data[i]
			count1 +=1
		if cluster_id[i] == 2:
			sum2 += trans_data[i]
			count2 +=1
	if count1 == 0:
		count1 += 1
	if count2 == 0:
		count2 += 1
	trans_mean1 = sum1/count1
	trans_mean2 = sum2/count2
	if trans_mean1 == prev_trans_mean1 and trans_mean2 == prev_trans_mean2:
		break
	else:
		prev_trans_mean2 = trans_mean2
		prev_trans_mean1 = trans_mean1
	count +=1

for i in range(len(x_val)):
	if cluster_id[i] == 1:
		plt.scatter(x_val[i],y_val[i], marker='+', c = 'red')
	else:
		plt.scatter(x_val[i],y_val[i], marker='o', c = 'green')
plt.show()


















