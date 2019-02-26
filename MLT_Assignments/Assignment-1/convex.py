import numpy as np
from numpy import linalg
#####################################loading the data##################################################
X_seen = np.load("data/AwA_python/X_seen.npy",encoding='latin1');
Xtest = np.load("data/AwA_python/Xtest.npy",encoding='latin1');
Ytest = np.load("data/AwA_python/Ytest.npy",encoding='latin1');
class_attributes_seen = np.load("data/AwA_python/class_attributes_seen.npy",encoding='latin1');
class_attributes_unseen = np.load("data/AwA_python/class_attributes_unseen.npy",encoding='latin1');
# print(X_seen[0].shape)
# print(X_seen[1].shape)


#####################################computing mean of each seen class###################################
#brr stores the means for all the seen classes
#brr[i] for i in (0,39) stores mean for (i+1)th class

brr = np.zeros((40,4096)) 
for i in range(0,40):
	brr[i] = np.mean(X_seen[i],axis=0)
# print(brr)


######################################computing the similarity##############################################
#arr stores the similarity among the class_attributes vectors for seen and unseen classes
#arr[i][j] stores the similarity among class_attributes vectors the ith seen class and jth unseen class
arr = np.zeros((40,10))
for i in range(0,40):
	for j in range(0,10):
		arr[i][j] = np.dot(class_attributes_seen[i], class_attributes_unseen[j])


#####################################normalizing the similarity vector#####################################
#normalizing the similarity vector for all the 10 unseen classes
#ith column of arr stores the normalized similarity scores of (i+1)th unseen class with the seen classes
for i in range(0,10):
	sum=0
	for j in range(0,40):
		sum+=arr[j][i]
	# print(sum)
	arr[:,i]=arr[:,i]/sum
# print(arr)


####################################computing means for unseen classes#####################################
#drr is just the transpose of arr to facilitate matrix multiplication
#crr stores the means of all the 10 unseen classes
#each row of crr corresponds to mean computed for an unseen class
#ith row corresponds to mean for (i+1)th unseen class
drr = np.transpose(arr)
# print(drr)
crr=np.matmul(drr,brr)
# print(crr)


####################################computing predicted labels for test examples#####################################
#err stores the predicted labels for all the 6180 test examples
#err[i] stores the predicted label for (i+1)th test example
err = np.zeros((6180,1))

for i in range(0,6180):
	#dist stores the eucledian distance computed for mean of first unseen class and feature vector of (i+1)th test example
	dist = np.linalg.norm(Xtest[i] - crr[0])
	err[i]=1
	for j in range(1,10):
		#temp stores the eucledian distance computed for mean of unseen classes varying between 2 to 10 and feature vector of (i+1)th test example. If temp is less than dist, then update the dist value and predicted label
		temp = np.linalg.norm(Xtest[i]-crr[j])
		if (temp < dist):
			dist=temp
			err[i]=j+1


####################################calculating number of correct labels and hence accuracy#####################################
#count stores the number of correctly predicted labels
count = 0
for i in range(0,6180):
	if(err[i] == Ytest[i]):
		count+=1
#printing the test-set classification accuracy for the model
print((count/6180)*100)
