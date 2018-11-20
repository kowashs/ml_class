#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

class Kernel:
    def __init__(self, X, ker_name='poly', hyper=None):
        self.X = X
        self.set_ker(ker_name,hyper)


    def set_ker(self,ker_name,hyper=None):
        if ker_name == 'poly':
            self.name = 'poly'
            n = len(self.X)
            self.base_func = lambda x1,x2: 1 + x1*x2
            self.base_mat = np.ones((n,n)) + np.outer(self.X,self.X) #1-d only
            self.hyper = hyper if hyper is not None else 0

        elif ker_name == 'rbf':
            self.name = 'rbf'
            n = len(self.X)
            XX = np.repeat([self.X],n,axis=0)
            Xdiff = XX - XX.T
            self.base_func = lambda x1,x2: np.exp(-1*np.dot(x1-x2,np.transpose(x1-x2)))
            self.base_mat = np.exp(-1*np.power(Xdiff,2)) #works for 1-d data
            self.hyper = (hyper if hyper is not None 
                          else 1/np.median(np.abs(Xdiff[np.triu_indices(m,1)])**2))
        
        else:
            print("Bad kernel name.")
            sys.exit()
    
    def get_matrix(self):
        return np.power(self.base_mat,self.hyper)

    def get_func(self):
        return lambda x1,x2: np.power(self.base_func(x1,x2),self.hyper)
    
    def get_predictor(self,w):
        func = self.get_func()
        return lambda x: np.dot(w, np.array([func(xi,x) for xi in self.X]))



def train_ker_ridge(ker_mat,Y_train,lam):
    n = ker_mat.shape[0]
    return np.linalg.solve(ker_mat+lam*np.identity(n),Y_train)



def loocv_lam(X, Y, hyper, ker_name='poly'):
    lams = 10.**np.arange(-8,8)
    val_errs = np.zeros_like(lams)
    
    for i in range(len(lams)):
        print(f"On lambda={lams[i]:2f}")
        x_inds = np.arange(len(X))

        errs = np.zeros_like(x_inds)

        for ind in x_inds:
            X_train = X[x_inds != ind]
            Y_train = Y[x_inds != ind]
            
            ker = Kernel(X_train,ker_name,hyper)
            ker_mat = ker.get_matrix()

            X_val = X[ind]
            Y_val = Y[ind]

            w = train_ker_ridge(ker_mat, Y_train, lams[i])
            predictor = ker.get_predictor(w)
            errs[ind] = np.abs(predictor(X_val)-Y_val)**2

        val_errs[i] = np.mean(errs)

    return lams, val_errs


def loocv_d(X, Y, dmin, dmax, lam, ker_name='poly'):
    ds = np.arange(dmin,dmax+1)
    val_errs = np.zeros_like(ds)

    for i in range(len(ds)):
        print(f"On d={ds[i]}")
        x_inds = np.arange(len(X))

        errs = np.zeros_like(x_inds)

        for ind in x_inds:
            X_train = X[x_inds != ind]
            Y_train = Y[x_inds != ind]
            
            ker = Kernel(X_train,ker_name,ds[i])
            ker_mat = ker.get_matrix()

            X_val = X[ind]
            Y_val = Y[ind]

            w = train_ker_ridge(ker_mat, Y_train, lam)
            predictor = ker.get_predictor(w)
            errs[ind] = np.abs(predictor(X_val)-Y_val)**2
        
        val_errs[i] = np.mean(errs)

    return ds, val_errs

def loocv_gam(X, Y, gmin, gmax, num_gs, lam, ker_name='rbf'):
    gams = np.linspace(gmin,gmax,num_gs)
    val_errs = np.zeros_like(gams)

    for i in range(len(gams)):
        print(f"On gamma={gams[i]}")
        x_inds = np.arange(len(X))

        errs = np.zeros_like(x_inds)
        
        for ind in x_inds:
            X_train = X[x_inds != ind]
            Y_train = Y[x_inds != ind]
            
            ker = Kernel(X_train,ker_name,gams[i])
            ker_mat = ker.get_matrix()

            X_val = X[ind]
            Y_val = Y[ind]

            w = train_ker_ridge(ker_mat, Y_train, lam)
            predictor = ker.get_predictor(w)
            errs[ind] = np.abs(predictor(X_val)-Y_val)**2
        
        val_errs[i] = np.mean(errs)
    
    return gams, val_errs




f = lambda x: 4*np.sin(np.pi*x)*np.cos(6*np.pi*np.power(x,2))
n = 30
X = np.random.uniform(0,1,n)
Y = f(X) + np.random.randn(n)



# ds = []
# lams = []
# d_curr = 0
# for it in range(10):
#     lams_cv,lam_errs = loocv_lam(X, Y, d_curr, 'poly')
#     lam_curr = lams_cv[np.argmin(lam_errs)]
#     ds_cv,d_errs = loocv_d(X, Y, 0, 40, lam_curr, 'poly')
#     d_curr = ds_cv[np.argmin(d_errs)]
#     
#     plt.plot(lams_cv,lam_errs)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.show()
#     plt.plot(ds_cv,d_errs)
#     plt.yscale('log')
#     plt.show()
# 
#     ds.append(d_curr)
#     lams.append(lam_curr)
# 
# 
# 
# print(ds,lams)
# 
# ker = Kernel(X,'poly',ds[-1])
# ker_mat = ker.get_matrix()
# 
# 
# x = np.linspace(0,1,1000)
# w = train_ker_ridge(ker_mat, Y, lams[-1])
# predictor = ker.get_predictor(w)
# 
# y_pred = np.array([predictor(xx) for xx in x])
# 
# plt.plot(x,y_pred,'-')
# plt.plot(X,Y,'o')
# plt.plot(x,f(x),'-')
# plt.show()



gams = []
lams = []
diffs = [np.abs(X[i]-X[j])**2 for i in range(len(X)-1) for j in range(i+1,len(X))]
gam_guess = 1/np.median(diffs)
gam_curr = gam_guess
for it in range(5):
    lams_cv,lam_errs = loocv_lam(X, Y, gam_curr, 'rbf')
    lam_curr = lams_cv[np.argmin(lam_errs)]
    gams_cv, gam_errs = loocv_gam(X,Y,0,10*gam_guess,100,lam_curr,'rbf')
    gam_curr = gams_cv[np.argmin(gam_errs)]

    plt.plot(lams_cv,lam_errs)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.plot(gams_cv,gam_errs)
    plt.yscale('log')
    plt.show()

    gams.append(gam_curr)
    lams.append(lam_curr)

print(gams,lams)
ker = Kernel(X,'rbf',gams[-1])
ker_mat = ker.get_matrix()

x = np.linspace(0,1,1000)
w = train_ker_ridge(ker_mat, Y, lams[-1])
predictor = ker.get_predictor(w)

y_pred = np.array([predictor(xx) for xx in x])

plt.plot(x,y_pred,'-')
plt.plot(X,Y,'o')
plt.plot(x,f(x),'-')
plt.show()


