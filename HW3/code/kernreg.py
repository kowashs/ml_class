#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
            self.base_func = lambda x1,x2: np.exp(-1*(x1-x2)**2)
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

def loocv(X, Y, params, ker_name='poly'):
    val_errs = np.zeros(len(params))

    for i,param in enumerate(params):
        print(f"On parameter set {i}")
        x_inds = np.arange(len(X))

        errs = np.zeros(len(x_inds))

        for ind in x_inds:
            X_train = X[x_inds != ind]
            Y_train = Y[x_inds != ind]

            ker = Kernel(X_train,ker_name,param[1])
            ker_mat = ker.get_matrix()

            X_val = X[ind]
            Y_val = Y[ind]

            w = train_ker_ridge(ker_mat, Y_train, param[0])
            predictor = ker.get_predictor(w)
            errs[ind] = np.abs(predictor(X_val)-Y_val)**2

        val_errs[i] = np.mean(errs)

    return val_errs



f = lambda x: 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2))
n = 30
X = np.random.uniform(0,1,n)
Y = f(X) + np.random.randn(n)


# params = np.array([(10**np.random.uniform(-7,1),float(np.random.uniform(0,50)))
#                    for _ in range(5000)])
# val_errs = loocv(X, Y, params, 'rbf')
# print(f"{params[np.argmin(val_errs)]}, {np.min(val_errs)}")

# ker = Kernel(X,'poly',18)
# ker_mat = ker.get_matrix()
# 
# w = train_ker_ridge(ker_mat,Y, 2.66e-7)
# predictor = ker.get_predictor(w)
# 
# x = np.linspace(0,1,1000)
# y = np.array([predictor(xx) for xx in x])
# 
# plt.plot(X,Y,'o',ms=8)
# plt.plot(x,f(x),'-',lw=2,label='f_true')
# plt.plot(x,y,'-',lw=2,label='f_pred')
# plt.xlim((0,1))
# plt.ylim((-6,6))
# plt.legend(fontsize=16)
# #plt.savefig('../figures/poly_ker_d20_lm4.pdf',bbox_inches='tight')
# plt.show()





ker = Kernel(X,'rbf',25.82)
ker_mat = ker.get_matrix()

w=train_ker_ridge(ker_mat,Y,4.94e-4)
predictor = ker.get_predictor(w)

x = np.linspace(0,1,1000)
y = np.array([predictor(xx) for xx in x])

plt.plot(X,Y,'o',ms=8)
plt.plot(x,f(x),'-',lw=2,label='f_true')
plt.plot(x,y,'-',lw=2,label='f_pred')
plt.xlim((0,1))
plt.ylim((-6,6))
plt.legend(fontsize=16)
# plt.savefig('../figures/rbf_ker_gp5_lm4.pdf',bbox_inches='tight')
plt.show()





# lams,val_errs = loocv_lam(X,Y,20,'poly')
# plt.plot(lams,val_errs,'o-')
# plt.xscale('log')
# plt.show()
# 
# 
# val_errs = loocv_hyper(X,Y,np.arange(0,30,2),1e-4,'poly')
# plt.plot(np.arange(0,30,2),val_errs,'o-')
# plt.show()
