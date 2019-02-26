import numpy as np
import matplotlib.pyplot as plt
import math
data = np.genfromtxt('kmeans_data.txt')

x_val = data[:,0]
y_val = data[:,1]
x_mean1 = data[0,0]
y_mean1 = data[0,1]
x_mean2 = data[1,0]
y_mean2 = data[1,1]

cluster_id = np.zeros(len(x_val))

l = np.random.randint(len(x_val)-1)
l_x = x_val[l]
l_y = y_val[l]

landmark = np.zeros(len(x_val))
for i in range(len(x_val)):
	landmark[i] = np.exp(-0.1 * (np.square(x_val[i]-l_x) + np.square(y_val[i]-l_y)))

prev_xmean1 = 0
prev_xmean2 = 0
prev_ymean1 = 0
prev_ymean2 = 0
count = 0
while count <= 50:
	a = np.exp(-0.1 * (np.square(x_mean1-l_x) + np.square(y_mean1-l_y)))
	b = np.exp(-0.1 * (np.square(x_mean2-l_x) + np.square(y_mean2-l_y)))

	for i in range(len(x_val)):
		dist1 = np.absolute(landmark[i]-a)
		dist2 = np.absolute(landmark[i]-b)
		if dist1 < dist2:
			cluster_id[i] = 1
		else:
			cluster_id[i] = 2
	xsum1 = 0
	ysum1 = 0
	xsum2 = 0
	ysum2 = 0
	count1 = 0
	count2 = 0
	for i in range(len(x_val)):
		if cluster_id[i] == 1:
			xsum1 += x_val[i]
			count1 +=1
		if cluster_id[i] == 2:
			xsum2 += x_val[i]
			count2 +=1
	if count1 == 0:
		count1 += 1
	if count2 == 0:
		count2 += 1
	x_mean1 = xsum1/count1
	x_mean2 = xsum2/count2
	count1 = 0
	count2 = 0
	for i in range(len(y_val)):
		if cluster_id[i] == 1:
			ysum1 += y_val[i]
			count1 +=1
		if cluster_id[i] == 2:
			ysum2 += y_val[i]
			count2 +=1
	if count1 == 0:
		count1 += 1
	if count2 == 0:
		count2 += 1
	y_mean1 = ysum1/count1
	y_mean2 = ysum2/count2

	if x_mean1 == prev_xmean1 and x_mean2 == prev_xmean2 and y_mean1 == prev_ymean1 and y_mean2== prev_ymean2:
		break
	else:
		prev_xmean1 = x_mean1
		prev_xmean2 = x_mean2
		prev_ymean1 = y_mean1
		prev_ymean2 = y_mean2
	count += 1
for i in range(len(x_val)):
	if cluster_id[i] == 1:
		plt.scatter(x_val[i],y_val[i], marker='+', c = 'red')
	else:
		plt.scatter(x_val[i],y_val[i], marker='o', c = 'green')
plt.scatter(l_x,l_y, marker='o', c = 'blue')
plt.show()





