#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mnist import MNIST
from multiprocessing import Pool
import os
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

def get_cluster(mu,X):
    return np.argmin([np.dot(X-mu[i],X-mu[i]) for i in range(k)])

def cluster_err(X,mu,cluster):
    dev = X[cluster,:] - mu[cluster]
    return np.sum(dev**2)

def assign(X,mu):
    n,d = X.shape
    k = len(mu)
    assignments = {i: [] for i in range(k)}

    with Pool(os.cpu_count()-1) as pool:
        assign_inds = np.array(pool.map(partial(get_cluster,mu), X))
   
    for i in range(k): assignments[i] = np.arange(n)[np.where(assign_inds == i)]

    return assignments

def recenter(X,assignments):
    k = len(assignments)
    n,d = X.shape

    mu = [np.mean(X[assignments[i],:],axis=0) for i in assignments]

    return mu

def objective(X,mu,assignments):
    k = len(mu)
    err = np.zeros(k)

#     with Pool(os.cpu_count()-1) as pool:
#         err = np.array(pool.map(partial(cluster_err,X,mu), np.arange(k)))


    for i in range(k):
        dev = X[assignments[i],:] - mu[i]
        dev *= dev

        err[i] = np.sum(dev)
  
    return np.sum(err)

if __name__ == '__main__':
    X_train, labels_train = load_train()
    X_test, labels_test = load_test()
    
    X = np.vstack((X_train,X_test))
    n,d = X.shape
    print("Data loaded.")
    
    k = int(sys.argv[1])
    thresh = 1e-3
    
    
    seeds = np.random.permutation(n)[:k]
    mu = X[seeds]
    mu_last = np.copy(mu)
    
    assignments = assign(X,mu)
    assignments_last = assignments.copy()
    print("Clusters seeded.")
    
    mu = recenter(X,assignments)
    print("First recentering")
    
    objs = [objective(X,mu,assignments)]
    i = 0
    
    print("Entering loop.")
    while np.max(np.abs(mu-mu_last)) > thresh:
        i += 1
        print(f"On iteration {i:4d}, last delta: {np.max(np.abs(mu-mu_last)):4f}")
    
        mu_last = np.copy(mu)
        assignments_last = assignments.copy()
    
        assignments = assign(X,mu)
        mu = recenter(X,assignments)
    
        objs.append(objective(X,mu,assignments))
    
    
        
    for i in range(k):
        plt.imshow(mu[i].reshape(28,28),aspect='equal',cmap=plt.get_cmap('binary'))
        plt.tight_layout()
        plt.savefig(f'../figures/{k}-clusters-{i}.pdf')
    
    plt.cla()
    plt.clf()
    
    plt.plot(np.log10(objs),'o',ms=3)
    plt.xlabel('Iteration', size=18)
    plt.ylabel('log(objective)', size=18)
    
    plt.tight_layout()
    plt.savefig(f'../figures/{k}-clusters_objective.pdf')

