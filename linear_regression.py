#fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality
import random as r
import numpy as np
import matplotlib.pyplot as plt 

def class_sep(n):
	if n<100:
		return 1
	elif n>100 and n<200:
		return 2
	

dataset = []

with open('winequality-white.csv', 'r') as fp:
	for line in fp:
		dataset.append(map(float, line.strip('\n').split(';')))

# Train
random.seed(1234)

w = np.array().reshape[1,num_weights]
i = 0

np.random.shuffle(dataset)

iteration = 0

while(iteration<10):
	print 'Iteration: ',iteration 
	np.random.shuffle(dataset)
	for data in dataset[:1000]:
		for i in range(11):
			calc += w[i]*data[i]
			
			continue
		else:
			i += 1
			print 'misclassified data points: ', data[0], data[1], data[2], data[3]
			x1.append(data[0])
			x2.append(data[1])
			x3.append(data[2])
			x4.append(data[3])
			w1 += data[0]*data[4]
			w2 += data[1]*data[4]
			w3 += data[2]*data[4]
			w4 += data[3]*data[4]
			print 'Updated weights: ', w1, w2, w3, w4
	iteration += 1

print('Updated weights after training: '+str(w1)+' '+str(w2)+' '+str(w3)+' '+str(w4))

E_in = 0.0
print 'Number of misclassified data points: ',i
E_in = float(i)*100/(60)
print 'E_in: ',E_in		


# Test
j = 0
for data in dataset[60:150]:
	calc = w1*data[0] + w2*data[1] + w3*data[2] + w4*data[3]
	if sign(calc) == data[4]:
		continue
	else:
		j += 1
		print('wrong weights')

E_out = 0.0
print 'Number of mis-classified data points in testing data: ',j
E_out = float(j)*100/(90)
print 'E_out: ',E_out

for data in dataset[0:150]:
	x1.append(data[0])

for data in dataset[0:150]:
	x2.append(data[1])

for data in dataset[0:150]:
	x3.append(data[2])

for data in dataset[0:150]:
	x4.append(data[3])

plt.subplot(221)
plt.plot(x1)

plt.subplot(222)
plt.plot(x2)

plt.subplot(223)
plt.plot(x3)

plt.subplot(224)
plt.plot(x4)

plt.show()
