#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


#synth lasso
data = np.loadtxt('data/synth-18_59_45')
lams,num_feats,tprs,fdrs = [data[:,i] for i in range(4)]
r = np.around(lams[-1]/lams[-2],2)

plt.plot(lams, num_feats, 'o')
plt.xscale('log')
plt.xlabel('lambda',size=16)
plt.ylabel('num_feats',size=16)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(f'../figures/synth_nfeats_{int(100*r)}.pdf')
plt.cla()




plt.plot(fdrs, tprs, 'o')
plt.xlabel('fdr',size=16)
plt.ylabel('tpr',size=16)
plt.tight_layout()
plt.savefig(f'../figures/synth_fdr-tpr_{int(100*r)}.pdf')
plt.cla()





#yelp lasso
data = np.loadtxt('data/yelp_sqrt-19_57_15')
lams,num_feats,ves,tes = [data[:,i] for i in range(4)]
r = np.around(lams[-1]/lams[-2],2)

plt.plot(lams,num_feats,'o')
plt.xscale('log')
plt.xlabel('lambda',size=16)
plt.ylabel('num_feats',size=16)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('../figures/yelp_nfeats.pdf')
plt.cla()

plt.plot(lams,ves/1000,'o',label='err_val')
plt.plot(lams,tes/4000,'o',label='err_train')
plt.xscale('log')
plt.xlabel('lambda',size=16)
plt.ylabel('MSE',size=16)
plt.gca().invert_xaxis()
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/yelp_errs.pdf')
plt.cla()




#gd
data = np.load('data/gdescent-23_22_40.npz')
j_train,j_test,e_train,e_test = [data[i][1:] for i in data.files]

plt.plot(j_train,'-o',label='j_train')
plt.plot(j_test,'-o',label='j_test')
plt.xlabel('iteration',size=16)
plt.ylabel('objective function',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_gd_obj.pdf')
plt.cla()



plt.plot(e_train,'-o',label='err_train')
plt.plot(e_test,'-o',label='err_test')
plt.xlabel('iteration',size=16)
plt.ylabel('error',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_gd_err.pdf')
plt.cla()



#sgd
data = np.load('data/sgdescent-1-23_22_17.npz')
j_train,j_test,e_train,e_test = [data[i][1:] for i in data.files]


plt.plot(j_train,'-o',label='j_train')
plt.plot(j_test,'-o',label='j_test')
plt.xlabel('iteration',size=16)
plt.ylabel('objective function',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_sgd_1_obj.pdf')
plt.cla()



plt.plot(e_train,'-o',label='err_train')
plt.plot(e_test,'-o',label='err_test')
plt.xlabel('iteration',size=16)
plt.ylabel('error',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_sgd_1_err.pdf')
plt.cla()




data = np.load('data/sgdescent-100-23_21_47.npz')
j_train,j_test,e_train,e_test = [data[i][1:] for i in data.files]

plt.plot(j_train,'-o',label='j_train')
plt.plot(j_test,'-o',label='j_test')
plt.xlabel('iteration',size=16)
plt.ylabel('objective function',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_sgd_100_obj.pdf')
plt.cla()



plt.plot(e_train,'-o',label='err_train')
plt.plot(e_test,'-o',label='err_test')
plt.xlabel('iteration',size=16)
plt.ylabel('error',size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('../figures/mnist_sgd_100_err.pdf')
plt.cla()
