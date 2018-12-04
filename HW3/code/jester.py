#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def parse_raw(raw):
    n_users = int(raw[-1,0])
    n_jokes = 100

    users = raw[:,0].astype(int)-1
    jokes = raw[:,1].astype(int)-1

    result = sp.coo_matrix((raw[:,2], (users, jokes))).tocsc()
    means = np.array(result.sum(axis=0)/result.getnnz(axis=0)).flatten()
    demeaned = sp.coo_matrix((raw[:,2]-means[jokes], (users,jokes)))


    return result, demeaned.tocsc(), means


def mse(R_pred,test_raw):
    n_data = len(test_raw)
    users = test_raw[:,0].astype(int)-1
    jokes = test_raw[:,1].astype(int)-1
    errs = R_pred[users,jokes] - test_raw[:,2]

    return np.sum(errs**2)/n_data

def mae(R_pred,test,test_raw):
    n_users,n_jokes = test.shape
    n_ratings = test.getnnz(axis=1)

    users = test_raw[:,0].astype(int)-1
    jokes = test_raw[:,1].astype(int)-1
    
    err = 0
    for i in range(n_users):
        ind = np.where(users == i)
        err_i = R_pred[users[ind],jokes[ind]] - test_raw[ind,2]
        err += np.sum(np.abs(err_i))/n_ratings[i]

    return err/n_users


def train_dumb(train):
    n_users,n_jokes = train.shape
    return (np.ones((n_users,1)),
            np.array(train.sum(axis=0)/train.getnnz(axis=0)))


def train_svd(train,d):
    U,S,Vt = svds(train,k=d)
    return U, sp.diags(S,0).dot(Vt)

def pred(U,Vt,means):
    n = U.shape[0]
    return U.dot(Vt) + np.outer(np.ones(n), means)



train_raw = np.genfromtxt('data/jester/train.txt',delimiter=',')
test_raw = np.genfromtxt('data/jester/test.txt',delimiter=',')

                           
                           

train, d_train, means_train = parse_raw(train_raw)
test, d_test, means_test = parse_raw(test_raw)

# U,Vt = train_dumb(d_train)
# R_pred = pred(U,Vt,means_train)

ds = [1,2,5,10,20,50]
mses_train = []
mses_test = []
maes_train = []
maes_test = []


for d in ds:
    print(f"On d={d}")
    U,Vt = train_svd(d_train,d)
    R_pred = pred(U,Vt,means_train)

    mses_test.append(mse(R_pred,test_raw))
    maes_test.append(mae(R_pred,test,test_raw))
    mses_train.append(mse(R_pred,train_raw))
    maes_train.append(mae(R_pred,train,train_raw))

ds = np.array(ds)
mses_train = np.array(mses_train)
maes_train = np.array(maes_train)
mses_test = np.array(mses_test)
maes_test = np.array(maes_test)
np.savez('data/svd.npz',d=ds,mse_train=mses_train,mae_train=maes_train,
         mse_test=mses_test,mae_test=maes_test)
