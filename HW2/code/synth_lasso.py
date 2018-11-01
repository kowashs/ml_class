#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from lasso import *


n = 500
d = 1000
k = 100
sig = 1

w_true = np.append(np.arange(1,k+1)/k, np.zeros(d-k))

X_train = np.random.randn(n,d)
Y_train = np.matmul(X_train, np.transpose(w_true)) + np.random.randn(n)*sig

lam_max = np.max(2*np.abs(np.matmul(Y_train - np.mean(Y_train), X_train)))


lams = []
num_feats = []
tprs = []
fdrs = []
xdrs = []
lam = lam_max
r = .75

while (max(num_feats) if num_feats else 0) < d:
    lams.append(lam)
    w = lasso_descend(X_train, Y_train, np.zeros(d), lam, 1e-5)[0]

    total_feats = np.count_nonzero(w)
    true_feats = np.count_nonzero(np.logical_and(w != 0, w_true != 0))
    false_feats = np.count_nonzero(np.logical_and(w != 0, w_true == 0))

    if total_feats != (true_feats + false_feats):
        print("Something is terribly wrong.")
    
    num_feats.append(total_feats)
    tprs.append(true_feats/k)
    fdrs.append(false_feats/total_feats if total_feats != 0 else 0)
    lam *= r 

lams = np.array(lams)
num_feats = np.array(num_feats)
tprs = np.array(tprs)
fdrs = np.array(fdrs)

plt.plot(lams, num_feats, 'o-')
plt.xscale('log')
plt.show()

plt.plot(lams, tprs, 'o-')
plt.plot(lams, fdrs, 'o-')
plt.xscale('log')
plt.show()



plt.plot(fdrs, tprs, 'o-')
plt.show()
