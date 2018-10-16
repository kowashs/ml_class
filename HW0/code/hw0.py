#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

n = 40000

#gaussian cdf
Z = np.random.randn(n)
plt.step(sorted(Z),np.arange(1,n+1)/n,label="Gaussian")


#y^(k) cdfs
for k in [1,8,64,512]:
    Yk = np.sum(np.sign(np.random.randn(n,k))*np.sqrt(1/k),axis=1)
    plt.step(sorted(Yk),np.arange(1,n+1)/n,label=k)
    



#plot details
plt.xlim((-3,3))
plt.ylim((0,1))
plt.xlabel("Value",fontsize=16)
plt.ylabel("Cumulative probability",fontsize=16)
plt.legend(fontsize=14)


#uncomment if testing
#plt.show()

#comment out if testing
plt.tight_layout()
plt.savefig('../figures/cdfs.pdf')
