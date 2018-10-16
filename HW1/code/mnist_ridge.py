#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
import time

def load_dataset():
    mndata = MNIST('../../mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.
    X_test = X_test/255.

    return X_train, X_test, labels_train, labels_test


def train(X, Y, L): # X is (n,d), Y is (n,k), L is reg. constant
    if len(X) != len(Y):
        print("Unequal input lengths; what have you done?")
        return

    d,k,n = len(X[0]), len(Y[0]), len(X)
    
    ep1 = time.time()
    
    M = np.zeros((d,d))
    eps = [ep1]
    deltas = []
    for i in range(n):
        M += np.outer(X[i],X[i])
        eps.append(time.time())
        deltas.append(eps[-1]-eps[-2])

    ep2 = time.time()
    print("Computed M in %ss." %np.around(ep2-ep1,3))
    print("Average of %ss per step. Max %ss, min %ss." %(np.around(np.mean(deltas),3),np.around(np.max(deltas),3),np.around(np.min(deltas),3)))

    N = np.zeros((d,k))
    for i in range(n):
        N += np.outer(X[i],Y[i])

    ep3 = time.time()
    print("Computed N in %ss." %np.around(ep3-ep2,4))


    return np.linalg.solve(M+L*np.identity(d), N)


def predict(W,X): # W is (d,k), X is (m,d)
    m = len(X)
    labels = np.zeros(m)

    Y_pred = np.matmul(X, W)
    for i in range(m):
        labels[i] = np.argmax(Y_pred[i])

    return labels


# load data and labels
X_train, X_test, labels_train, labels_test = load_dataset()


# encode labels as unit vectors
codes = np.identity(10) 
Y_train = np.array([codes[labels_train[i]] for i in range(len(labels_train))])
Y_test = np.array([codes[labels_test[i]] for i in range(len(labels_test))])


W = train(X_train, Y_train, 1e-4)
labels_pred = predict(W, X_test)
error = np.count_nonzero(labels_pred - labels_test)/len(labels_pred)

print(error)
