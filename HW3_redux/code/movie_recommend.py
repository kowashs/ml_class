#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import sys
from multiprocessing import Pool

def parse_raw(raw):
    n_users = int(raw[-1,0])
    n_movies = 500

    users = raw[:,0].astype(int)-1
    movies = raw[:,1].astype(int)-1

    result = sp.coo_matrix((raw[:,2], (users, movies))).tocsc()
    means = np.array(result.sum(axis=0)/result.getnnz(axis=0)).flatten()
    demeaned = sp.coo_matrix((raw[:,2]-means[movies], (users,movies)))


    return result, demeaned.tocsc(), means


def mse(R_pred,test_raw):
    n_data = len(test_raw)
    users = test_raw[:,0].astype(int)-1
    movies = test_raw[:,1].astype(int)-1
    errs = R_pred[users,movies] - test_raw[:,2]

    return np.sum(errs**2)/n_data

def mae(R_pred,test,test_raw):
    n_users,n_movies = test.shape
    n_ratings = test.getnnz(axis=1)

    users = test_raw[:,0].astype(int)-1
    movies = test_raw[:,1].astype(int)-1
    
    err = 0
    for i in range(n_users):
        ind = np.where(users == i)
        err_i = R_pred[users[ind],movies[ind]] - test_raw[ind,2]
        err += np.sum(np.abs(err_i))/n_ratings[i]

    return err/n_users


def train_dumb(train):
    n_users,n_movies = train.shape
    return (np.ones((n_users,1)),
            np.array(train.sum(axis=0)/train.getnnz(axis=0)))


def train_svd(train,d):
    U,S,Vt = svds(train,k=d)
    return U, sp.diags(S,0).dot(Vt)

def train_clever(train,train_raw,d,lam,thresh=1e-1):
    train_r = train.tocsr()
    n_users,n_movies = train.shape
    users = range(n_users)
    movies = range(n_movies)

    mov_rated = [train_r[user].nonzero()[1] for user in users]
    usr_rated = [train[:,movie].nonzero()[0] for movie in movies]


    u_old = np.random.randn(n_users, d)
    v_old = np.random.randn(n_movies,d)
    u = np.zeros_like(u_old)
    v = np.zeros_like(v_old)
    delta_u = 1000*np.ones_like(u_old)
    delta_v = 1000*np.ones_like(v_old)
    i = 1

    while (np.max(delta_u) > thresh) or (np.max(delta_v) > thresh):
        u,v = u_old.copy(), v_old.copy()

        for user in users:
            v_red = v[mov_rated[user]]
            rhs = train_r[user].dot(v).reshape(d)
            lh_mat = v_red.T @ v_red + lam*np.eye(d)

            u[user] = np.linalg.solve(lh_mat, rhs)



        for movie in movies:
            u_red = u[usr_rated[movie]]
            rhs = (train[:,movie].T).dot(u).reshape(d)
            lh_mat = u_red.T @ u_red + lam*np.eye(d)
            
            v[movie] = np.linalg.solve(lh_mat, rhs)


        delta_u = np.abs(u-u_old)
        delta_v = np.abs(v-v_old)
        u_old,v_old = u.copy(),v.copy()
        print(f"Step {i}: du={np.max(delta_u)}, dv={np.max(delta_v)}", end='\r')
        i += 1

    print("\n")
    return u, v.T

def pred(U,Vt,means):
    n = U.shape[0]
    return U.dot(Vt) + np.outer(np.ones(n), means)



train_raw = np.genfromtxt('data/train.txt',delimiter=',')
test_raw = np.genfromtxt('data/test.txt',delimiter=',')

train, d_train, means_train = parse_raw(train_raw)
test, d_test, means_test = parse_raw(test_raw)

if len(sys.argv) < 2:
    print("Not enough args.")
    sys.exit()

if sys.argv[1] == 'svd':
    ds = np.array([1, 2, 5, 10, 20, 50])
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

    mses_train = np.array(mses_train)
    maes_train = np.array(maes_train)
    mses_test = np.array(mses_test)
    maes_test = np.array(maes_test)
    np.savez('data/svd.npz',d=ds,mse_train=mses_train,mae_train=maes_train,
             mse_test=mses_test,mae_test=maes_test)




elif sys.argv[1] == 'clever':
    ds = np.array([1,2,5,10,20,50])
    lams = 10.**np.arange(-7,2)
    for lam in lams:
        mses_train = []
        mses_test = []
        maes_train = []
        maes_test = []
        print(f"On lam={lam}")
        for d in ds:
            print(f"On d={d}")
            U,Vt = train_clever(d_train,train_raw,d,lam)
            R_pred = pred(U,Vt,means_train)
            
        
            mses_test.append(mse(R_pred,test_raw))
            maes_test.append(mae(R_pred,test,test_raw))
            mses_train.append(mse(R_pred,train_raw))
            maes_train.append(mae(R_pred,train,train_raw))

        mses_train = np.array(mses_train)
        maes_train = np.array(maes_train)
        mses_test = np.array(mses_test)
        maes_test = np.array(maes_test)
        np.savez(f'data/clever_{round(np.log10(lam))}.npz',d=ds,
                 mse_train=mses_train,mae_train=maes_train,mse_test=mses_test,
                 mae_test=maes_test)
    

elif sys.argv[1] == 'dumb':
    U,Vt = train_dumb(d_train)
    R_pred = pred(U,Vt,means_train)

    print(mse(R_pred,train_raw))
    print(mae(R_pred,train,train_raw))
    print(mse(R_pred,test_raw))
    print(mae(R_pred,test,test_raw))









