#!/usr/bin/env python
import numpy as np
from mnist import MNIST
import sys

def load_test():
    mndata = MNIST('../../mnist/data/')
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_test = X_test/255.

    return X_test, labels_test

def load_train():
    mndata = MNIST('../../mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_train = X_train/255.

    return X_train, labels_train

def demean(X): #array must have more than one axis!
    mu = np.mean(X,axis=0)
    return X-mu, mu

def pred_error(labels_pred, labels_true):
    if len(labels_pred) != len(labels_true):
        print("Unequal label list lengths.")
        return
    
    misses = np.count_nonzero(labels_pred-labels_true)
    total = len(labels_true)
    print("Missed %s out of %s, relative error %s." %(misses,total,misses/total))

    return misses/total


def train(X, Y, L): # X is (n,d), Y is (n,k), L is reg. constant
    if len(X) != len(Y):
        print("Unequal input lengths; what have you done?")
        return

    d,k,n = len(X[0]), len(Y[0]), len(X)
    
    Xt = np.transpose(X)

    M = np.dot(Xt,X)
    N = np.dot(Xt,Y)

    return np.linalg.solve(M+L*np.identity(d), N)


def predict(W,X,Y_mu=0): # W is (d,k), X is (m,d), Y_mu is (1,k)
    m = len(X)

    Y_pred = np.dot(X, W) + Y_mu

    return np.argmax(Y_pred, axis=1)



# load data and labels
X_train, labels_train = load_train()
X_test, labels_test = load_test()


# encode labels as unit vectors
codes = np.identity(10) 
Y_train = np.array([codes[labels_train[i]] 
                       for i in range(len(labels_train))])

X_train, X_mu = demean(X_train)
Y_train, Y_mu = demean(Y_train)

W = train(X_train, Y_train, 1e-4)

labels_pred = predict(W, X_test - X_mu, Y_mu)
print("Test error is %3f." %pred_error(labels_pred,labels_test))
