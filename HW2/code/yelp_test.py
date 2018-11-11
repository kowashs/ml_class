#!/usr/bin/env python
# yelp_test.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from lasso import *


    


# Load data according to provided example
X = np.genfromtxt("data/upvote_data.csv", delimiter=",")
Y = np.loadtxt("data/upvote_labels.txt", dtype=np.int)
feature_names = open("data/upvote_features.txt").read().splitlines()

print("Data loaded.")

d = X.shape[1]

X_train = X[:4000]
Y_train = np.sqrt(Y[:4000])

X_val = X[4000:5000]
Y_val = np.sqrt(Y[4000:5000])

X_test = X[5000:]
Y_test = np.sqrt(Y[5000:])


lam = 1.11

w,b = lasso_descend(X_train, Y_train, np.zeros(d),lam,5e-2)

test_pred = np.matmul(X_test,w) + b
val_pred = np.matmul(X_val,w) + b
train_pred = np.matmul(X_train,w) + b

test_diff = test_pred - Y_test
val_diff = val_pred - Y_val
train_diff = train_pred - Y_train

test_err = np.matmul(test_diff,test_diff)
val_err = np.matmul(val_diff,val_diff)
train_err = np.matmul(train_diff,train_diff)

top_feats_ind = np.argsort(np.abs(w))[-10:]
top_weights = w[top_feats_ind]
top_labels = [feature_names[i] for i in top_feats_ind]


with open('data/topfeats','w') as out:
    out.write(f'Train error: {train_err:4f}\n')
    out.write(f'Val error: {val_err:4f}\n')
    out.write(f'Test error: {test_err:4f}\n')
    out.write('\n\n')
    out.write(f'{"Feature":16s} {"Weight":8s}\n')
    out.write('-'*25+'\n')
    out.writelines([f'{top_labels[i]:<16s}|{top_weights[i]: 8f}\n'
                    for i in range(len(top_labels))])

