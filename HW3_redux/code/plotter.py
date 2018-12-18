#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# svd recommend
data = np.load('data/svd.npz')
d = data['d']
ms_tr,ms_te = data['mse_train'],data['mse_test']
ma_tr,ma_te = data['mae_train'],data['mae_test']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(d,ms_tr,'o-',ms=5,lw=2,label='mse_train')
ax.plot(d,ms_te,'o-',ms=5,lw=2,label='mse_test')
ax.legend(fontsize=18)
ax.tick_params(axis='both',labelsize=16)
ax.set_xlim((0,55))
ax.set_ylim((0.3,0.6))
fig.savefig('../figures/svd_mse.pdf',bbox_inches = 0)
plt.cla()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(d,ma_tr,'o-',ms=5,lw=2,label='mae_train')
ax.plot(d,ma_te,'o-',ms=5,lw=2,label='mae_test')
ax.legend(fontsize=18)
ax.tick_params(axis='both',labelsize=16)
ax.set_xlim((0,55))
ax.set_ylim((0.3,0.6))
fig.savefig('../figures/svd_mae.pdf',bbox_inches = 0)
plt.cla()





# clever recommend
for lam in ['-70','-60','-50','-40','-30','-20','-10','00','10']:
    data = np.load(f'data/clever_{lam}.npz')
    d = data['d']
    ms_tr,ms_te = data['mse_train'],data['mse_test']
    ma_tr,ma_te = data['mae_train'],data['mae_test']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d,ms_tr,'o-',ms=5,lw=2,label='mse_train')
    ax.plot(d,ms_te,'o-',ms=5,lw=2,label='mse_test')
    ax.legend(fontsize=18)
    ax.tick_params(axis='both',labelsize=16)
    fig.savefig(f'../figures/clever_{lam}_mse.pdf',bbox_inches = 0)
    plt.cla()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d,ma_tr,'o-',ms=5,lw=2,label='mae_train')
    ax.plot(d,ma_te,'o-',ms=5,lw=2,label='mae_test')
    ax.legend(fontsize=18)
    ax.tick_params(axis='both',labelsize=16)
    fig.savefig(f'../figures/clever_{lam}_mae.pdf',bbox_inches = 0)
    plt.cla()
