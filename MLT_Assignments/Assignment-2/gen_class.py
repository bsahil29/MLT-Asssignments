import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import LinearSVC
x1 = []
x2 = []
y_value = []
with open('binclass.txt', 'r') as f:
    content = f.readlines()
    for x in content:
        row = x.split(',')
        x1.append(row[0])
        x2.append(row[1])
        y_value.append(row[2].strip())
arr1 = np.column_stack((x1,x2));        
arr = np.column_stack((x1,x2,y_value));
# print(type(arr))
# print(arr)
pos=[]
neg=[]
for i in arr:
	if i[2]=='1':
		pos.append(i)
	else:
		neg.append(i)
pos=np.array(pos).astype(np.float)
neg=np.array(neg).astype(np.float)
pos_mean = np.mean(pos, axis=0)
# print(pos_mean)
neg_mean = np.mean(neg, axis=0)
# print(neg_mean.shape)
pos1 = np.delete(pos, -1, axis=1)
neg1 = np.delete(neg, -1, axis=1)
pos_mean1 = np.delete(pos_mean, -1, axis=0)
neg_mean1 = np.delete(neg_mean, -1, axis=0)
pos_deviation = pos1 - pos_mean1
neg_deviation = neg1 - neg_mean1
# print(pos_deviation.shape)
pos_sum=0 
for i in pos_deviation:
    pos_sum=pos_sum+np.matmul(i.transpose(),i)
pos_variance=pos_sum/400
neg_sum=0 
for i in neg_deviation:
    neg_sum=neg_sum+np.matmul(i.transpose(),i)
neg_variance=neg_sum/400
# print(pos_variance)
# print(neg_variance)
plt.scatter(pos1[:,0], pos1[:,1], marker='+', c = 'red')
plt.scatter(neg1[:,0], neg1[:,1], marker='o', c = 'blue')
# plt.show()
x_axis = np.linspace(-15,35,1000)
y_axis = np.linspace(-5,45,1000)
x,y = np.meshgrid(x_axis,y_axis)
# print(x_axis)
# print(x_axis.ravel())
# print(x)
# print(x.ravel())
plt.contour(x_axis,y_axis.ravel(),((1/(2*pos_variance))*((x-pos_mean1[0])*(x-pos_mean1[0])+(y-pos_mean1[1])*(y-pos_mean1[1])))-((1/(2*neg_variance))*((x-neg_mean1[0])*(x-neg_mean1[0])+(y-neg_mean1[1])*(y-neg_mean1[1])))-math.log(neg_variance/pos_variance),[0])
# plt.show()

pos_variance = neg_variance
plt.contour(x_axis,y_axis.ravel(),((1/(2*pos_variance))*((x-pos_mean1[0])*(x-pos_mean1[0])+(y-pos_mean1[1])*(y-pos_mean1[1])))-((1/(2*neg_variance))*((x-neg_mean1[0])*(x-neg_mean1[0])+(y-neg_mean1[1])*(y-neg_mean1[1])))-math.log(neg_variance/pos_variance),[0])
# plt.show()

# clf = LinearSVC(dual=False)
# clf.fit(arr1, y_value)
# plt.contour(x_axis,y_axis.ravel(),clf.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape))
plt.show()
