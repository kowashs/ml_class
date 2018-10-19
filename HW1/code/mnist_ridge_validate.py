#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
import sys
import time

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

def cos_tx(p,d):
    var = 0.1
    Gt = np.random.randn(d,p)*np.sqrt(var)
    b = np.random.uniform(0,2*np.pi,p)
    

    return lambda X: np.cos(np.dot(X,Gt) + b)


#command-line signature: mnist_ridge.py p
p = int(sys.argv[1])



# load data and labels
X_train_tot, labels_train_tot = load_train()
# X_test, labels_test = load_test()

# encode labels as unit vectors
codes = np.identity(10) 
Y_train_tot = np.array([codes[labels_train_tot[i]] 
                       for i in range(len(labels_train_tot))])
# Y_test = np.array([codes[labels_test[i]] for i in range(len(labels_test))])



# partition data into train/val
idx = np.random.permutation(60000)
ind_train = idx[0:48000]
ind_val = idx[48000:]

X_train = X_train_tot[ind_train]
Y_train = Y_train_tot[ind_train]
labels_train = labels_train_tot[ind_train]

X_val = X_train_tot[ind_val]
Y_val = Y_train_tot[ind_val]
labels_val = labels_train_tot[ind_val]






# map features
feature_tx = cos_tx(p, len(X_train[0]))
H_train = feature_tx(X_train)
H_val = feature_tx(X_val)
# H_test = feature_tx(X_test)

# H_train, H_mu = demean(H_train)
# Y_train, Y_mu = demean(Y_train)

print("Created feature matrix of shape %s" %str(H_train.shape))

# train classifier on de-meaned features
W = train(H_train, Y_train, 1e-4)
print("Trained classifier of shape %s with Frobenius norm %s" %(str(W.shape), np.trace(np.matmul(W.T, W))))


# predict to measure train/validation/test error
labels_train_pred = predict(W, H_train) 
labels_val_pred = predict(W, H_val)
# labels_test_pred = predict(W, H_test)

train_error = pred_error(labels_train_pred, labels_train)
val_error = pred_error(labels_val_pred, labels_val)
# test_error = pred_error(labels_test_pred, labels_test)


with open(f'data/{p}','a') as f:
    f.write('%s,%s,%s\n' %(p,train_error,val_error))
