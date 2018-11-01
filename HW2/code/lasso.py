#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def descend_step(X, Y, w_curr, lam):
    d = X.shape[1]
    w = w_curr
    b = np.mean(Y - np.matmul(X,np.transpose(w)))

    for k in range(d):
        slice_ind = np.where(np.arange(d) != k)[0]
        a = 2*np.matmul(X[:,k], X[:,k])
        c = 2*np.matmul(X[:,k], Y - b - np.matmul(X[:,slice_ind], np.transpose(w[slice_ind])))

        if np.abs(c) <= lam:
            w[k] = 0
        else:
            w[k] = (c - np.sign(c)*lam)/a

    return w



def lasso_descend(X, Y, w_init, lam, thresh):
    n = X.shape[0]
    d = X.shape[1]

    if len(w_init) != d:
        print("Initial guess dimension mismatch. Setting all zeros.")
        w_init = np.zeros(d)
    

    #do at least one step
    w_last = w_init
    w_curr = descend_step(X, Y, w_init, lam)

    while np.max(np.abs(w_curr-w_last)) > thresh:
        w_last = w_curr
        w_curr = descend_step(X, Y, w_curr, lam)

    b = np.mean(Y - np.matmul(X,np.transpose(w_curr)))

    return w_curr, b






