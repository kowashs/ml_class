#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import timeit


def cos_tx(p,d):
    var = 0.1
    G = np.random.randn(p,d)*np.sqrt(var)
    b = np.random.uniform(0,2*np.pi,p)

    return lambda X: np.dot(X,np.transpose(G)) + b

X = np.random.randn(10000,784)

times = {}
for p in range(10,30):
    feature_tx = cos_tx(p,784)
    wrapped = lambda: feature_tx(X)
    times[p] = timeit.timeit(wrapped, number=100)/100

print(times)
