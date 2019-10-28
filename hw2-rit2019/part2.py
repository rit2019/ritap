#!/usr/bin/env python
# coding: utf-8

# Generate circle dataset

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

# Generate 100 noisy circle data points
n_samples = 100


# You can uncomment the following line to plot these circle data points
# plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')

##################################
#sys.stdout = open("part2.txt","w+")

for i in np.arange(0,6):
    (X, y) = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=0)
    kmeans = KMeans(n_clusters=3,random_state=i)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    print(i)
    print(centers)

#sys.stdout.close()
##################################







##################################

