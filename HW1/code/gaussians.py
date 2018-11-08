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
    print(mu_samp)
    sig_samp = np.matmul(np.transpose(X-mu_samp),X-mu_samp)/(n-1)
    
    evals, evecs = np.linalg.eigh(sig_samp)
    print(evecs)
    
    e1 = evals[np.argmax(evals)]
    u1 = evecs[:,np.argmax(evals)]

    e2 = evals[np.argmin(evals)]
    u2 = evecs[:,np.argmin(evals)]
    
    ax.set_xlim((-7,7))
    ax.set_ylim((-7,7))

    ax.plot(X[:,0],X[:,1],'^')
    ax.plot([mu_samp[0],mu_samp[0]+np.sqrt(evals[0])*evecs[0,0]],[mu_samp[1],mu_samp[1]+np.sqrt(evals[0])*evecs[1,0]],lw=5)
    ax.plot([mu_samp[0],mu_samp[0]+np.sqrt(evals[1])*evecs[0,1]],[mu_samp[1],mu_samp[1]+np.sqrt(evals[1])*evecs[1,1]],lw=5)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('../figures/unscale_plot_%s.pdf' %(i+1))
    ax.cla()

    X_tilde = np.matmul(X-mu_samp,np.transpose([u1,u2]))/np.sqrt([e1,e2])
    ax.plot(X_tilde[:,0],X_tilde[:,1],'o',ms=6)
    
    ax.set_xlim((-7,7))
    ax.set_ylim((-7,7))

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('../figures/scale_plot_%s.pdf' %(i+1))

