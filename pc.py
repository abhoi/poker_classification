import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Preprocessing
# Load training data
X_train_list = []
with open('poker_train.txt', 'r') as inputfile:
	for row in csv.reader(inputfile):
		X_train_list.append(row)

# Convert X_train from list to np_array
X_train = np.asarray(X_train_list)

print "X_train shape: ", X_train.shape

# Extract class labels
y_train = X_train[:, [10]]
y_train = y_train.reshape(25010)
y_train = y_train.astype(np.int32)

print "y_train shape: ", y_train.shape
#print y[0, 0:6]

# Delete class labels from X_train
X_train = X_train[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

# Scale training data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Convert X_train from float64 to int32
X_train = X_train.astype(np.int32)

print "X_train shape: ", X_train.shape
#print r[0:6]


# MLP Classification
clf = MLPClassifier(solver='adam', alpha=0.00001, 
	hidden_layer_sizes=(225, 4), 
	max_iter=300, learning_rate='adaptive', verbose=True, tol=1e-8)
slf = svm.SVC(decision_function_shape='ovo', verbose=True)
linslf = svm.LinearSVC(verbose=True)
print clf.get_params(deep=True)
print slf.get_params(deep=True)
print linslf.get_params(deep=True)

clf.fit(X_train, y_train)
slf.fit(X_train, y_train)
linslf.fit(X_train, y_train)


# Load testing data
X_test_list = []
with open('poker_test.txt', 'r') as inputfile:
	for row in csv.reader(inputfile):
		X_test_list.append(row)

# Convert X_test from list to np_array
X_test = np.asarray(X_test_list)

print "X_test shape: ", X_test.shape

# Extract class labels
y_test = X_test[:, [10]]
y_test = y_test.reshape(1000000)
y_test = y_test.astype(np.int32)

print "y_test shape: ", y_test.shape

# Delete class labels from X_test
X_test = X_test[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
X_test = X_test.astype(np.int32)

print "X_test shape: ", X_test.shape

# Score X_test
print "clf: ", clf.score(X_test, y_test)
print "slf: ", slf.score(X_test, y_test)
print "linslf: ", linslf.score(X_test, y_test)