#!/usr/bin/env python
# lasso.py
import numpy as np
import matplotlib.pyplot as plt


def descend_step(X, Y, w_curr, lam):
    d = X.shape[1]
    w = w_curr
    b = np.mean(Y - np.matmul(X,np.transpose(w)))
    
    xks = {k: X[:,k] for k in range(d)}
    a = {k: 2*np.matmul(xks[k],xks[k]) for k in range(d)}

    for k in range(d):
        wk = np.copy(w)
        xk = xks[k]
        wk[k] = 0 #faster way to knock out a column

        c = 2*np.matmul(xk, Y - b - np.matmul(X, np.transpose(wk)))

        if np.abs(c) <= lam:
            w[k] = 0
        else:
            w[k] = (c - np.sign(c)*lam)/a[k]

    return w



def lasso_descend(X, Y, w_init, lam, thresh):
    n = X.shape[0]
    d = X.shape[1]

    if len(w_init) != d:
        print("Initial guess dimension mismatch. Setting all zeros.")
        w_init = np.zeros(d)
    
    
    #do at least one step
    w_last = np.copy(w_init)
    w_curr = descend_step(X, Y, w_init, lam)
    
    i=0
    while np.max(np.abs(w_curr-w_last)) > thresh:
        i += 1
        print("on descent step %s" %i)
        w_last = np.copy(w_curr)
        w_curr = descend_step(X, Y, w_curr, lam)
        print(np.max(np.abs(w_curr-w_last)))

    b = np.mean(Y - np.matmul(X,np.transpose(w_curr)))

    return w_curr, b






