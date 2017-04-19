import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import os.path
import time

# Preprocessing
#np.set_printoptions(threshold=np.nan)

# Load training data
X_train_list = []
with open('poker_train.txt', 'r') as inputfile:
	for row in csv.reader(inputfile):
		X_train_list.append(row)

# Convert X_train from list to np_array
X_train = np.asarray(X_train_list)

#print "X_train shape: ", X_train.shape

# Extract class labels
y_train = X_train[:, [10]]
y_train = y_train.reshape(25010)
y_train = y_train.astype(np.int32)

#print "y_train shape: ", y_train.shape
#print y[0, 0:6]

# Delete class labels from X_train
X_train = X_train[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

# Scale training data
"""scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)"""

new_suit = np.zeros(4)
new_rank = np.zeros(13)
new_hand = np.zeros((85, ))
new_X_train = np.zeros((25010, 85))
new_y_train = np.zeros((25010, 10))
new_X_test = np.zeros((1000000, 85))
new_y_test = np.zeros((1000000, 10))

print new_suit.shape, new_suit
print new_rank.shape, new_rank
print new_hand.shape

X_train = X_train.astype(np.int32)

for i in range(0, 25010):
	#print "/New card/"
	k = 0
	for j in range(0, 10):
		if j % 2 == 0:
			new_suit = np.zeros(4)
			#print "Suit: ", X_train[i, j]
			new_suit[X_train[i, j] - 1] = 1
			#print new_suit
			new_hand[k:k+4] = new_suit
			k = k + 4
		else:
			new_rank = np.zeros(13)
			#print "Rank: ", X_train[i, j]
			new_rank[X_train[i, j] - 1] = 1
			#print new_rank
			new_hand[k:k+13] = new_rank
			k = k + 13 
	new_X_train[i, :] = new_hand
#print new_X_train[25008:25010, :]

print "new_X_train shape: ", new_X_train.shape

y_train = y_train.astype(np.int32)

for i in range(0, 25010):
	new_class = np.zeros(10)
	new_class[y_train[i] - 1] = 1
	#print new_class
	new_y_train[i, :] = new_class 
#print new_y_train[:, 25008:25010]

print "new_y_train shape: ", new_y_train.shape

# Convert X_train from float64 to int32
X_train = X_train.astype(np.int32)

#print "X_train shape: ", X_train.shape
#print r[0:6]


# MLP Classification
print "Fitting data..."

if os.path.isfile("mlpclassifier.pkl"):
	print "clf found"
	clf = joblib.load('mlpclassifier.pkl')
else:
	print "clf not found"
	clf = MLPClassifier(solver='adam', alpha=0.0001, 
	hidden_layer_sizes=(85, 4),
	max_iter=300, verbose=True, warm_start=True,
	learning_rate='adaptive', tol=1e-8)
	print clf.get_params(deep=True)
	clf.fit(new_X_train, new_y_train)
	joblib.dump(clf, 'mlpclassifier.pkl')
	# hidden_layer_sizes=(65,2) worked best (97.8522) or (170, 4)

# Load testing data
X_test_list = [] 
with open('poker_test.txt', 'r') as inputfile:
	for row in csv.reader(inputfile):
		X_test_list.append(row)

# Convert X_test from list to np_array
X_test = np.asarray(X_test_list)

#print "X_test shape: ", X_test.shape

# Extract class labels
y_test = X_test[:, [10]]
y_test = y_test.reshape(1000000)
y_test = y_test.astype(np.int32)

for i in range(0, 1000000):
	new_class = np.zeros(10)
	new_class[y_test[i] - 1] = 1
	#print new_class
	new_y_test[i, :] = new_class

print "new_y_test shape: ", new_y_test.shape

# Delete class labels from X_test
X_test = X_test[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
X_test = X_test.astype(np.int32)

for i in range(0, 1000000):
	#print "/New card/"
	k = 0
	for j in range(0, 10):
		if j % 2 == 0:
			new_suit = np.zeros(4)
			#print "Suit: ", X_train[i, j]
			new_suit[X_test[i, j] - 1] = 1
			#print new_suit
			new_hand[k:k+4] = new_suit
			k = k + 4
		else:
			new_rank = np.zeros(13)
			#print "Rank: ", X_train[i, j]
			new_rank[X_test[i, j] - 1] = 1
			#print new_rank
			new_hand[k:k+13] = new_rank
			k = k + 13 
	new_X_test[i, :] = new_hand

print "new_X_test shape: ", new_X_test.shape

# Score X_test
start = time.time()
print "NN: ", clf.score(new_X_test, new_y_test) * 100
end = time.time()
print (end - start)