#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp
from sp.linalg import svds

train_raw = np.genfromtxt('data/jester/train.txt')
test_raw = np.genfromtxt('data/jester/test.txt')

def raw_to_sparse(raw):
    n_users = int(raw[-1,0])
    n_jokes = 100

    result = sp.coo_matrix((raw[:,2],(raw[:,0],raw[:,1])), shape=(n_users,n_jokes))
    return result.to_csr()

def mse_slow(U,V,test):
    errs = (np.dot(U,V) - R)**2
    errs /= len(errs)jk
    return np.sum(errs)


n_users = int(train_raw[-1,0])
n_jokes = 100

train = raw_to_sparse(train_raw)


