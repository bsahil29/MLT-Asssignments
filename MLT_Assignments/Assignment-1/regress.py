import numpy as np
from numpy import linalg
#####################################loading the data##################################################
X_seen = np.load("data/AwA_python/X_seen.npy",encoding='latin1');
Xtest = np.load("data/AwA_python/Xtest.npy",encoding='latin1');
Ytest = np.load("data/AwA_python/Ytest.npy",encoding='latin1');
class_attributes_seen = np.load("data/AwA_python/class_attributes_seen.npy",encoding='latin1');
class_attributes_unseen = np.load("data/AwA_python/class_attributes_unseen.npy",encoding='latin1');
arr = np.zeros((40,10))
# print(X_seen[0].shape)
# print(X_seen[1].shape)


#####################################computing mean of each seen class###################################
#brr stores the means for all the seen classes
#brr[i] for i in (0,39) stores mean for (i+1)th class
brr = np.zeros((40,4096))
for i in range(0,40):
	brr[i] = np.mean(X_seen[i],axis=0)

#computing transpose of class_attributes_seen which will be used for computing W later
class_attributes_seen_transpose=np.transpose(class_attributes_seen)

#my_list stores the lambda values for which we have to test the accuracy of our model
my_list = [0.01, 0.1, 1, 10, 20, 50, 100]
#length of my_list
my_list_len=len(my_list)

#max_accuracy stores the maximum accuracy for the values of lambda among my_list
#max_lambda stores the value of lambda for which maximum accuracy is attained among the lambda values from my_list
max_accuracy=0
max_lambda= 0.01
#each iteration of for loop computes the test-set classification accuracy for one value of lambda i.e. my_list[i]
for i in range(0,my_list_len):
	lamda = my_list[i]
	#arr is the diagonal matrix whose all diagonal enteries are equal to lambda i.e. λI
	arr = lamda * np.identity(85);
	# print(arr*lamda)

	#crr stores the product of matrix multiplication of class_attributes_seen and its transpose i.e. (As.transpose * As)
	crr = np.matmul(class_attributes_seen_transpose,class_attributes_seen)
	#drr stores the inverse computed for sum of crr and arr i.e. (As.transpose * As + λI).inverse
	drr = np.linalg.inv(crr+arr)
	#err stores the product of inverse (drr) computed with class_attributes_seen_transpose i.e. (As.transpose * As + λI).inverse * As.transpose 
	err= np.matmul(drr,class_attributes_seen_transpose)
	#W stores the matrix of weights that was to be learned i.e (As.transpose * As + λI).inverse * As.transpose * Ms
	W= np.matmul(err,brr)
	# print(W.shape)

	####################################computing means for unseen classes#########################################################
	#means stores the means of all the 10 unseen classes
	#each row of means corresponds to mean computed for an unseen class
	#ith row corresponds to mean for (i+1)th unseen class
	means = np.matmul(class_attributes_unseen,W)
	# print(means.shape)

	####################################computing predicted labels for test examples#####################################
	#prediction stores the predicted labels for all the 6180 test examples
	#prediction[i] stores the predicted label for (i+1)th test example
	prediction = np.zeros((6180,1))

	for i in range(0,6180):
		#dist stores the eucledian distance computed for mean of first unseen class and feature vector of (i+1)th test example
		dist = np.linalg.norm(Xtest[i] - means[0])
		prediction[i]=1
		for j in range(1,10):
			#temp stores the eucledian distance computed for mean of unseen classes varying between 2 to 10 and feature vector of (i+1)th test example. If temp is less than dist, then update the dist value and predicted label
			temp = np.linalg.norm(Xtest[i]-means[j])
			if (temp < dist):
				dist=temp
				prediction[i]=j+1

	####################################calculating number of correct labels and hence accuracy#####################################
	#count stores the number of correctly predicted labels
	count = 0
	for i in range(0,6180):
		if(prediction[i] == Ytest[i]):
			count+=1

	#printing the test-set classification accuracy for the model for value of lambda = lamda
	print(lamda,(count/6180)*100)
	
	#if accuracy for current value of lambda is greater than max_accuracy, update the max_accuracy and max_lambda
	if ((count/6180)*100 > max_accuracy):
		max_accuracy = (count/6180)*100
		max_lambda = lamda

#printing the test-set classification accuracy and lambda value for max_accuracy and max_lambda
print(max_lambda, max_accuracy)




