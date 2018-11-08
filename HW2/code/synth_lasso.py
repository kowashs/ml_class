#!/usr/bin/env python
# synth_lasso.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from lasso import *





n = 500
d = 1000
k = 100
sig = 1

w_true = np.append(np.arange(1,k+1)/k, np.zeros(d-k))

X_train = np.random.randn(n,d)
Y_train = np.matmul(X_train, np.transpose(w_true)) + np.random.randn(n)*sig

lam_max = np.max(2*np.abs(np.matmul(Y_train - np.mean(Y_train), X_train)))


lams = []
ws = []
num_feats = []
tprs = []
fdrs = []
xdrs = []
lam = lam_max

r = float(sys.argv[1])


it = 0
while (max(num_feats) if num_feats else 0) < d:
    it += 1
    print(f"On iter {it} with {num_feats[-1] if num_feats else 0} "
          f"features and lambda={lam:.5f}")
    lams.append(lam)
    w = lasso_descend(X_train, Y_train, ws[-1] if ws else np.zeros(d),
                      lam, 1e-3)[0]

    total_feats = np.count_nonzero(w)
    true_feats = np.count_nonzero(np.logical_and(w != 0, w_true != 0))
    false_feats = np.count_nonzero(np.logical_and(w != 0, w_true == 0))

    if total_feats != (true_feats + false_feats):
        print("Something is terribly wrong.")
    
    num_feats.append(total_feats)
    tprs.append(true_feats/k)
    fdrs.append(false_feats/total_feats if total_feats != 0 else 0)
    ws.append(w)
    lam *= r 

lams = np.array(lams)
num_feats = np.array(num_feats)
tprs = np.array(tprs)
fdrs = np.array(fdrs)


ftime = datetime.now().time()
stamp = f"{ftime.hour:02d}_{ftime.minute:02d}_{ftime.second:02d}"
with open(f'data/synth-{stamp}','w') as f:
    f.writelines([f"{lams[i]:12e} {num_feats[i]:4d} "
                  f"{tprs[i]:14.8f} {fdrs[i]:14.8f}\n"
                  for i in range(len(lams))])
