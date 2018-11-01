#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from lasso import *


# Load data according to provided example
X = np.genfromtxt("upvote_data.csv", delimiter=",")
y = np.loadtxt("upvote_labels.txt", dtype=np.int)
feature_names = open("upvote_features.txt").read().splitlines()






