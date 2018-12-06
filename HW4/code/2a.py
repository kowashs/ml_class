#!/usr/bin/env python
import numpy as np
import cvxpy as cvx
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def f(x):
    if type(x) is np.ndarray:
        y = np.zeros_like(x)
        for i in range(4):
            y[np.where(x>=(i+1)/5)] += 1
    else:
        y = 0
        for i in range(4):
            if x >= (i+1)/5:
                y += 1
    
    return 10*y

def k_scal(x,y,gamma):
    return np.exp(-gamma*(x-y)**2)

def k_matrix(x,gamma):
    n = len(x)
    k = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            k[i,j] = np.exp(-gamma*(x[i]-x[j])**2)
    return k

def predict(x,alpha,gamma): #single scalars at a time
    return lambda y: np.dot(alpha, k_scal(x,y,gamma))

n = 50
x = np.arange(n)/(n-1)
y = f(x) + np.random.randn(n)
y[24] = 0

gams = np.random.uniform(0,60,500)
lambs = np.power(10,np.random.uniform(-7,0.5,500))
params = np.stack((gams,lambs)).T
errs = np.zeros(len(params))



for i in range(len(params)):
    print(f"On parameter set {i+1}")
    for omit in range(n):
        x_loo = np.concatenate((x[:omit],x[omit+1:]))
        y_loo = np.concatenate((y[:omit],y[omit+1:]))
        gamma = params[i,0]
        lambd = params[i,1]
    
        alpha = cvx.Variable(n-1)
    
        k_mat = k_matrix(x_loo,gamma)
    
        objective = cvx.Minimize(cvx.sum_squares(y_loo - k_mat*alpha)
                                 + lambd*cvx.quad_form(alpha,k_mat))
    
        prob = cvx.Problem(objective)
    
        prob.solve()
        
        alpha_sol = alpha.value
        pred = predict(x_loo, alpha_sol, gamma)
    
        errs[i] += (y[omit] - pred(x[omit]))**2
    
    errs[i] /= n


ftime = datetime.now().time()
stamp = f"{ftime.hour:02d}_{ftime.minute:02d}_{ftime.second:02d}"
with open(f"../data/2a-{stamp}",'w') as f:
    f.writelines([f"{params[i,0]:12.2f} {params[i,1]:12.2e} {errs[i]:12.4f}\n"
                  for i in range(len(params))])

