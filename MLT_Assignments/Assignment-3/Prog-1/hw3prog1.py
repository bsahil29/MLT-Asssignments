import numpy as np
import matplotlib.pyplot as plt
import random
train = np.genfromtxt('ridgetrain.txt')
test = np.genfromtxt('ridgetest.txt')
# print(train[0])
# print(test.size)
x_train = train[:,0]
# print(x_train.size)
y_train = train[:,1]
x_test = test[:,0]
y_test = test[:,1]
# plt.show()

kernel = np.zeros((len(x_train),len(x_train)))
for i in range(0,len(x_train)):
	for j in range(0, len(x_train)):
		kernel[i][j] = np.exp(-0.1 * np.square(x_train[i]-x_train[j]))

list1 = [0.1, 1, 10, 100]
for l in list1:
	arr = kernel + l*np.identity(len(x_train))
	brr = np.linalg.inv(arr)
	crr = np.matmul(brr,y_train)

	y_pred = np.zeros(len(y_test))
	for i in range(len(y_pred)):
		for j in range(len(x_train)):
			y_pred[i] += crr[j]*np.exp(-0.1 * np.square(x_test[i]-x_train[j]))
	plt.scatter(x_test,y_test, marker='+', c = 'blue')
	plt.scatter(x_test,y_pred, marker='o', c = 'red')
	plt.show()
	error = np.sqrt(((y_pred - y_test) ** 2).mean())
	print(l , " ", error)


list2 = [2, 5, 20, 50, 100]

for l1 in list2:
	new_x = np.zeros(l1)
	new_y = np.zeros(l1)
	# print(new_y.shape)
	for i in range(l1):
		val = random.randint(0,len(x_train)-1)
		new_x[i] = x_train[val]
		new_y[i] = y_train[val]
	landmark = np.zeros((len(x_train),l1))
	for i in range(len(x_train)):
		for j in range(l1):
			landmark[i][j] = np.exp(-0.1 * np.square(x_train[i]-new_x[j]))

	drr = np.matmul(landmark.transpose(),landmark) + (0.1*np.identity(l1))
	# print(drr.shape)
	err = np.linalg.inv(drr)
	# print(err.shape)
	frr = np.matmul(err,landmark.transpose())
	# print(frr.shape)
	grr = np.matmul(frr,y_train)

	y_pred1 = np.zeros(len(y_test))
	for i in range(len(y_pred)):
		for j in range(len(grr)):
			y_pred1[i] += grr[j]*np.exp(-0.1 * np.square(x_test[i]-new_x[j]))
	plt.scatter(x_test,y_test, marker='+', c = 'blue')
	plt.scatter(x_test,y_pred1, marker='o', c = 'red')
	plt.show()
	error1 = np.sqrt(((y_pred1 - y_test) ** 2).mean())
	print(l1 , " ", error1)

























