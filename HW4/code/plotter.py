#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt




#2a
data = np.genfromtxt('../data/2a-23_14_03')
fig = plt.figure()
ax = fig.add_subplot(111)
cf = ax.tricontourf(data[:,0],data[:,1],data[:,2],150)
plt.colorbar(cf)

plt.xlabel('gamma', size=16)
plt.ylabel('lambda', size=16)

plt.yscale('log')


ax.tick_params(axis='both',labelsize=12)
plt.savefig('../figures/2a_loocv.png',dpi=400,bbox_inches=0)
plt.cla()
plt.clf()













#2b
data = np.genfromtxt('../data/2b-23_35_41')
fig = plt.figure()
ax = fig.add_subplot(111)
cf = ax.tricontourf(data[:,0],data[:,1],data[:,2],200)
plt.colorbar(cf)

plt.xlabel('gamma', size=16)
plt.ylabel('lambda', size=16)

plt.yscale('log')


ax.tick_params(axis='both',labelsize=12)
plt.savefig('../figures/2b_loocv.png',dpi=400,bbox_inches=0)
plt.cla()
plt.clf()












#2c

















#3a




















#3b
fig = plt.figure()
ax = fig.add_subplot(111)
data = np.genfromtxt('../data/3b_r50_p50000_accs')

ns = np.arange(1,len(data)+1)
ax.plot(ns,data[:,0],'o-',ms=5,label='train')
ax.plot(ns,data[:,1],'o-',ms=5,label='test')
ax.set_xlabel('iteration', size=16)
ax.set_ylabel('accuracy',size=16)
ax.set_xticks(ns)
ax.legend(fontsize=16)
ax.tick_params(axis='both',labelsize=12)

plt.savefig('../figures/3b_acc.pdf',bbox_inches=0)
plt.cla()
plt.clf()





#3c
fig = plt.figure()
ax = fig.add_subplot(111)
data = np.genfromtxt('../data/3c_r10_p70000_M100_N9_accs')

ns = np.arange(1,len(data)+1)
ax.plot(ns,data[:,0],'o-',ms=5,label='train')
ax.plot(ns,data[:,1],'o-',ms=5,label='test')
ax.set_xlabel('iteration', size=16)
ax.set_ylabel('accuracy',size=16)
ax.set_xticks(ns)
ax.legend(fontsize=16)
ax.tick_params(axis='both',labelsize=12)

plt.savefig('../figures/3c_acc.pdf',bbox_inches=0)
plt.cla()
plt.clf()
