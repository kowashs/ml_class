#!/usr/bin/env python
# grad_descend.py
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from datetime import datetime
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

def grad_w(mu, X, Y, w, lam):
    n = len(mu)
    return np.dot((mu-1)*Y,X)/n + 2*lam*w

def grad_b(mu, X, Y, w, lam):
    n = len(mu)
    return np.dot(mu-1, Y)/n

def hess_w(mu, X, Y, w, lam):
    n = len(mu)
    d = X.shape[1]
    hess = np.zeros((d,d))
    for i in range(n):
        hess += mu[i]*(1-mu[i])*Y[i]**2*np.outer(X[i],X[i])
    
    return hess/n + 2*lam*np.identity(d)

def hess_b(mu, X, Y, w, lam):
    n = len(mu)
    
    return np.sum(mu*(1-mu)*Y**2)/n


def objective(mu, X, Y, w, lam):
    n = len(mu)
    
    return np.sum(-1*np.log(mu))/n + lam*np.matmul(w,w)


def pred(X, w, b):
    return np.sign(b+ np.matmul(X,w))


def gdescend(X_train, Y_train, X_test, Y_test, lam=0.1, eta = 0.4, tol = 1e-4):
    d = X_train.shape[1]
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    j_train = []
    j_test = []
    e_train = []
    e_test = []

    w = np.zeros(d)
    b = 0
    
    
    mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
    mu_test = 1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))
    j_train.append(objective(mu_train, X_train, Y_train, w, lam))
    j_test.append(objective(mu_test, X_test, Y_test, w, lam))
    e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
    e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    i=0
    print(f"Step {i}")
    
    
    while (len(j_train) < 2 or
           (np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0) > tol):
        i += 1
        print(f"Step {i}: delta = {(np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0)}")
        gb = grad_b(mu_train, X_train, Y_train, w, lam)
        gw = grad_w(mu_train, X_train, Y_train, w, lam)
        
        b -= eta*gb
        w -= eta*gw
        
        mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
        mu_test = 1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))
        
        j_train.append(objective(mu_train, X_train, Y_train, w, lam))
        j_test.append(objective(mu_test, X_test, Y_test, w, lam))
        
        e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
        e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    
    return j_train,j_test,e_train,e_test




def sgdescend(X_train, Y_train, X_test, Y_test, batch, lam=0.1, eta=0.4, tol=1e-4):
    d = X_train.shape[1]
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    j_train = []
    j_test = []
    e_train = []
    e_test = []

    w = np.zeros(d)
    b = 0
    
    

    mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
    mu_test = 1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))

    j_train.append(objective(mu_train, X_train, Y_train, w, lam))
    j_test.append(objective(mu_test, X_test, Y_test, w, lam))
    e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
    e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    i=0
    print(f"Step {i}")
    
    
    while (len(j_train) < 2 or
           (np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0) > tol):
        i += 1
        print(f"Step {i}: delta = {(np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0)}")

        batch_ind = np.random.randint(0,n_train,batch)
        X_batch = X_train[batch_ind]
        Y_batch = Y_train[batch_ind]
        mu_batch = 1/(1+np.exp(-1*Y_batch*(b + np.matmul(X_batch, w))))

        gb = grad_b(mu_batch, X_batch, Y_batch, w, lam)
        gw = grad_w(mu_batch, X_batch, Y_batch, w, lam)
        
        b -= eta*gb
        w -= eta*gw
        
        mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
        mu_test = 1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))
        
        j_train.append(objective(mu_train, X_train, Y_train, w, lam))
        j_test.append(objective(mu_test, X_test, Y_test, w, lam))
        
        e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
        e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    
    return j_train,j_test,e_train,e_test




def newt_descend(X_train, Y_train, X_test, Y_test, lam=0.1, eta=0.4, tol=1e-4):
    d = X_train.shape[1]
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    j_train = []
    j_test = []
    e_train = []
    e_test = []

    w = np.zeros(d)
    b = 0
    
    
    mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
    mu_test = 1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))
    j_train.append(objective(mu_train, X_train, Y_train, w, lam))
    j_test.append(objective(mu_test, X_test, Y_test, w, lam))
    e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
    e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    i=0
    print(f"Step {i}")

    while (len(j_train) < 2 or
           (np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0) > tol):
        i += 1
        print(f"Step {i}: delta = {(np.abs(j_train[-1]-j_train[-2]) if len(j_train) > 1 else 0)}")

        vw = np.linalg.solve(hess_w(mu_train, X_train, Y_train, w, lam),
                             -1*grad_w(mu_train, X_train, Y_train, w, lam))
        gb = grad_b(mu_train,X_train,Y_train,w,lam)
        hb = hess_b(mu_train,X_trian,Y_train,w,lam)
        vb = -1*gb/hb

        b += eta*vb
        w += eta*vw

        mu_train = 1/(1+np.exp(-1*Y_train*(b + np.matmul(X_train, w))))
        mu_test =  1/(1+np.exp(-1*Y_test*(b + np.matmul(X_test, w))))

        j_train.append(objective(mu_train, X_train, Y_train, w, lam))
        j_test.append(objective(mu_test, X_test, Y_test, w, lam))
        
        e_train.append(np.count_nonzero(pred(X_train, w, b) - Y_train)/n_train)
        e_test.append(np.count_nonzero(pred(X_test, w, b) - Y_test)/n_test)
    
    return j_train,j_test,e_train,e_test






X_train, labels_train = load_train()
X_test, labels_test = load_test()

train_twosev = np.where(np.logical_or(labels_train == 2, labels_train == 7))
test_twosev =  np.where(np.logical_or(labels_test == 2, labels_test == 7))


X_train = X_train[train_twosev]
labels_train = labels_train[train_twosev]

X_test = X_test[test_twosev]
labels_test = labels_test[test_twosev]

codes = {2: -1, 7: 1}
Y_train = np.array([codes[i] for i in labels_train])
Y_test = np.array([codes[i] for i in labels_test])



if sys.argv[1] == 'gd':
    j_train,j_test,e_train,e_test = gdescend(X_train, Y_train, X_test, Y_test,
                                             eta=float(sys.argv[2]))
    now = datetime.now().time()
    stamp = f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
    np.savez(f'data/gdescent-{stamp}', j_train=j_train, j_test=j_test,
             e_train=e_train, e_test=e_test)

elif sys.argv[1] == 'sgd':
    j_train,j_test,e_train,e_test = sgdescend(X_train, Y_train, X_test,
                                              Y_test, eta=float(sys.argv[2]),
                                              batch=int(sys.argv[3]))
    now = datetime.now().time()
    stamp = f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
    np.savez(f'data/sgdescent-{sys.argv[3]}-{stamp}', j_train=j_train,
             j_test=j_test, e_train=e_train, e_test=e_test)


elif sys.argv[1] == 'newt':
    j_train,j_test,e_train,e_test = newt_descend(X_train, Y_train, X_test,
                                                 Y_test, eta=float(sys.argv[2]))
    
    now = datetime.now().time()
    stamp = f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
    np.savez(f'data/newtdescent-{stamp}', j_train=j_train,
             j_test=j_test, e_train=e_train, e_test=e_test)



    
    





