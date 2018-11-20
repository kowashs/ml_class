#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys


def rbf_matrix(X,gamma):
    n = X.shape[0]
    K = np.zeros_like((n,n))

    for i in range(len(X)):
        for j in range(len(X)):
            K[i,j] = -1*gamma*np.dot(X[i]-X[j],X[i]-X[j])

    return np.exp(K)

def poly_matrix(X,d):
    n = X.shape[0]
    K = np.zeros_like((n,n))

    for i in range(len(X)):
        for j in range(len(X)):
            K[i,j] = 1 + np.dot(X[i],np.transpose(X[j]))

    return np.power(K,d)

def poly_func(d):
    return lambda x1,x2: np.power(1 + np.dot(x1,np.transpose(x2)), d)

def rbf_func(gamma):
    return lambda x1,x2: np.exp(-1*gamma*np.dot(x1-x2,np.transpose(x1-x2)))

def ker_predict(w,ker_func,X,x):
    ks = np.array([ker_func(X[i],x) for i in range(len(X))])
    return np.dot(w,ks)



def train_ker_ridge(ker_mat, Y_train, lam):
    n = ker_mat.shape[0]
    return np.linalg.solve(ker_mat+lam*np.identity(n),Y_train)

def loo(X,Y,inds,ind):
    return X[inds != ind],Y[inds != ind]




def cv_poly(X, Y, lmin, lmax, n_lam, d_min, d_max, n_iters):
    lams = []
    ds = []
    
    d_curr = 0
    lam_curr = 1
    for it in range(n_iters):
        x_inds = np.arange(len(X))
        lam_errs = np.zeros_like(x_inds)

        for ind in x_inds:
            X_train, Y_train = loo(X,Y,x_inds,ind)
            X_val,Y_val = X[ind],Y[ind]

            ker_mat = poly_matrix(X_train, d_curr)
            w = train_ker_ridge(ker_mat, Y_train, lam)
            errs[ind] = np.abs(ker_predict(w,poly_func(d_curr),X_train,X_val) - Y_val)**2

        


