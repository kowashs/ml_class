#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lasso import *


    


# Load data according to provided example
X = np.genfromtxt("data/upvote_data.csv", delimiter=",")
Y = np.loadtxt("data/upvote_labels.txt", dtype=np.int)
feature_names = open("data/upvote_features.txt").read().splitlines()

print("Data loaded.")

d = X.shape[1]

X_train = X[:4000]
Y_train = np.sqrt(Y[:4000])

X_val = X[4000:5000]
Y_val = np.sqrt(Y[4000:5000])

X_test = X[5000:]
Y_test = np.sqrt(Y[5000:])


lam_max = np.max(2*np.abs(np.matmul(Y_train - np.mean(Y_train), X_train)))


lams = []
num_feats = []
val_errs = []
train_errs = []
ws = []
bs = []

lam = lam_max
r = .9

print("Entering loop.")
while (max(num_feats) if num_feats else 0) < .95*d:
    lams.append(lam)
    w,b = lasso_descend(X_train, Y_train, (ws[-1] if ws else np.zeros(d)), lam, 1e-2)
    ws.append(w)
    bs.append(b)


    val_pred = np.matmul(X_val, np.transpose(w)) + b
    train_pred = np.matmul(X_train, np.transpose(w)) + b

    val_diff = val_pred - Y_val 
    train_diff = train_pred - Y_train
    
    feats = np.count_nonzero(w)
    val_err = np.matmul(val_diff, val_diff)
    train_err = np.matmul(train_diff, train_diff)

    num_feats.append(feats)
    val_errs.append(val_err)
    train_errs.append(train_err)
#   print(f"{lam:12e} {feats:4d} {val_err:14.6f} {train_err:14.6f}")

    lam *= r



ftime = datetime.now().time()
stamp = f"{ftime.hour:02d}_{ftime.minute:02d}_{ftime.second:02d}"
with open(f"data/yelp_sqrt-{stamp}", "w") as f:
    f.writelines([f"{lams[i]:12e} {num_feats[i]:4d} {val_errs[i]:14.6f} {train_errs[i]:14.6f}\n" for i in range(len(lams))])













