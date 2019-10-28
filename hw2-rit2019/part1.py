#!/usr/bin/env python
# coding: utf-8

# # KMeans on "Iris" Dataset

from sklearn.datasets import load_digits
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
#import sys

iris = datasets.load_iris()
X = iris.data
y = iris.target

#################################
#     My Code Here 
##################################
#sys.stdout = open("part1.txt","w+") 


for i in np.arange(2,11):
    estimator = KMeans(n_clusters=i,random_state=0)
    estimator.fit(X)
    result=estimator.labels_
    print(i)
    print(np.bincount(result))

    
#sys.stdout.close()
##################################
