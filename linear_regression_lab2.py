#Rajat Dangi - 201451038
import numpy as np
import random
import time
import matplotlib.pyplot as plt 

f = open('winequality-white.csv', 'r') #data file

weights = []
data = []
x = []
y = []
leng = 4897 #length of data

for i in range(leng):
	data.append(map(float, f.readline().strip().split(';')))
	x.append(data[i][0:10]) #features
	y.append(data[i][11]) 	#ratings

#train-test set
train_set = []
train_rating = []
test_set = []
test_rating = []

E_in = E_out = 0
error_in = []
error_out = []
q=wtx=wt=i=j=k=l=0
iteration = 0 #Fractoins for 

for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:		#Training - Testing Partition
	loop = int(leng*frac)
	iteration = 1-frac
	for i in range(loop): 					#train set for diff. frac.
		train_set.append(x[i])
		train_rating.append(y[i])
	
	test_size = leng-loop

	for j in range(test_size):				#test set for diff. frac.
		test_set.append(x[j])
		test_rating.append(y[j])
	train_set_plus = np.linalg.pinv(train_set)		#Pseudo-inverse 7 weights calculation
	weights.append(map(float, train_set_plus.dot(train_rating)))
	
	for l in range(loop):					#Error in training data
		wtx = sum((np.dot(weights, train_set[l]) - train_rating[l]) * (np.dot(weights, train_set[l]) - train_rating[l]))
	E_in = wtx/(loop)
	print('E_in: ',E_in)
	error_in.append((frac, E_in))
	wtx = 0

	for k in range(test_size):				#Error in testing data
		wt = sum((np.dot(weights, test_set[k]) - test_rating[k]) * (np.dot(weights, test_set[k]) - test_rating[k]))
	E_out = wt/(test_size)
	print('E_out: ',E_out)
	error_out.append((iteration, E_out))
	wt = 0

#Plot

fig = plt.figure()
training_plot = fig.add_subplot(211)
testing_plot = fig.add_subplot(212)

training_plot.set_xlabel('Dataset Partition')
training_plot.set_ylabel('Error')

testing_plot.set_xlabel('Dataset Partition')
testing_plot.set_ylabel('Error')

testing_plot.set_title('Testing')
training_plot.set_title('Training')

fig.suptitle('Error in training and testing for different dataset size:')

training_plot.loglog(*zip(*error_in))
testing_plot.loglog(*zip(*error_in))
#plt.subplot(211)
#plt.plot(*zip(*error_in))

#plt.subplot(212)
#plt.plot(*zip(*error_out))

plt.savefig('linear_regression.png', dpi=300)