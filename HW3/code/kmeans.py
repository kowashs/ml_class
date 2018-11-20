#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
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



def kpp_init(k, X):
    n,d = X.shape
    
    mu = [X[np.random.randint(n)]] #choose first cluster uniformly
    print("Center 1 placed.")

    assignments = assign(X,mu)

    while len(mu) < k:
        prob = np.zeros(n)
        for assn in assignments:
            for j in assignments[assn]:
                prob[j] = np.dot(X[j]-mu[assn],X[j]-mu[assn])
        
        prob /= np.sum(prob)
        mu.append(X[np.random.choice(n,p=prob)])
        print(f"Center {len(mu)} placed.")
        assignments = assign(X,mu)

    return np.array(mu),assignments

def get_cluster(mu,X):
    k = len(mu)
    return np.argmin([np.dot(X-mu[i],X-mu[i]) for i in range(k)])

def assign(X,mu):
    n,d = X.shape
    k = len(mu)
    assignments = {i: [] for i in range(k)}
    
    assign_inds = np.fromiter(map(partial(get_cluster,mu),X),
                              dtype=int,count=n)
#     with Pool(os.cpu_count()-1) as pool:
#         assign_inds = np.fromiter(pool.map(partial(get_cluster,mu), X),
#                                   dtype=int,count=n)
   
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
    
    if sys.argv[2] == 'uni':
        seeds = np.random.permutation(n)[:k]
        mu = X[seeds]
        mu_last = np.copy(mu)
        
        assignments = assign(X,mu)
        assignments_last = assignments.copy()
        print("Clusters seeded.")
    elif sys.argv[2] == 'kpp':
        mu, assignments = kpp_init(k,X)
        mu_last = np.copy(mu)
        assignments_last = assignments.copy()
    else: print("Bad arg, specify uni or kpp")

    mu = recenter(X,assignments)
    print("First recentering")
    
    objs = [objective(X,mu,assignments)]
    i = 0
    
    assignments = assign(X,mu)
    mu = recenter(X,assignments)
    print("Manual first step.")

    
    print("Entering loop.")
    while np.any([np.any(assignments_last[i] != assignments[i]) for i in assignments]):
        i += 1
        print(f"On iteration {i:4d}, last delta: {np.max(np.abs(mu-mu_last)):4f}")
    
        mu_last = np.copy(mu)
        assignments_last = assignments.copy()
    
        assignments = assign(X,mu)
        mu = recenter(X,assignments)
    
        objs.append(objective(X,mu,assignments))
    
    print("Escaped the loop.")
    
    fname = f'{k}-clusters' if sys.argv[2] == 'uni' else f'{k}-clusterspp'    
    for i in range(k):
        plt.imshow(mu[i].reshape(28,28),aspect='equal',cmap=plt.get_cmap('binary'))
        plt.tight_layout()
        plt.savefig(f'../figures/{fname}-{i}.pdf')
    
    plt.cla()
    plt.clf()
    
    plt.plot(np.log10(objs),'o',ms=3)
    plt.xlabel('Iteration', size=18)
    plt.ylabel('log(objective)', size=18)
    
    plt.tight_layout()
    plt.savefig(f'../figures/{fname}_objective.pdf')
