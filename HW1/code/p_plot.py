#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

with open('run_final','r') as f:
    lines = f.readlines()
    p = np.array([line.split(',')[0] for line in lines], dtype=int)
    train_errs = np.array([line.split(',')[1] for line in lines], dtype=np.float64)
    val_errs = np.array([line.split(',')[2] for line in lines], dtype=np.float64)

sort_ind = np.argsort(p)
p = p[sort_ind]
train_errs = train_errs[sort_ind]
val_errs = val_errs[sort_ind]


plt.plot(np.log10(p),train_errs,lw=1.5,label='training error')
plt.plot(np.log10(p),val_errs,lw=1.5,label='validation error')

plt.xlim((-0.1,4.5))
plt.ylim((-0.05,1))

plt.legend(fontsize=14)
plt.xlabel(r'$\log(p)$', size=16)
plt.ylabel('error', size=16)
plt.tight_layout()
plt.savefig('../figures/val_plot.pdf')
