#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

mus = np.array([[1,2],[-1,1],[2,-2]])
sigmas = np.array([[[1,0],[0,2]],[[2,-1.8],[-1.8,2]],[[3,1],[1,2]]])

n = 100

for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mu = mus[i]
    sigma = sigmas[i]
    rt_sigma = la.sqrtm(sigma)
    s = np.linalg.det(sigma)

    Y = np.random.randn(n,2)
    X = np.transpose(np.matmul(rt_sigma,np.transpose(Y)))+mu
    

    mu_samp = np.sum(X,axis=0)/n
    sig_samp = np.matmul(np.transpose(X-mu_samp),X-mu_samp)/(n-1)
    
    evals, evecs = np.linalg.eigh(sig_samp)
    print(evals)
    
    ax.set_xlim((-10,10))
    ax.set_ylim((-10,10))

    ax.plot(X[:,0],X[:,1],'^')
    
    ax.set_aspect('equal')
    plt.show()
