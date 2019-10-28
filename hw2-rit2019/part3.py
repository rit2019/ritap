#!/usr/bin/env python
# coding: utf-8

# Digits Datset

import numpy as np
import seaborn as sns; sns.set()  # for plot styling
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import mode


# Load digits dataset
digits = datasets.load_digits()
X = digits.data
y_true = digits.target

# Uncomment the following lines to plot the first digit
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

def permute_labels(cluster_labels):
    """Finds best permutation between cluster labels and true labels.

    NOTE: Clustering doesn't have any information about the true
    labels so this just matches the indices of the cluster labels
    to the most likely true label.

    This makes the main diagonal of the confusion matrix to have
    the largest values.
    """
    permuted_labels = np.zeros_like(cluster_labels)
    for i in range(10):
        sel = (cluster_labels == i)
        permuted_labels[sel] = mode(y_true[sel])[0]
    return permuted_labels



################################## My Code ######################################
#import sys
#sys.stdout = open("part3.txt","w+")    

r=0
n = [1,5,10,50]

for p in list(n):
    model=KMeans(n_clusters=10,max_iter =p,random_state=0)
    model.fit(X)
    y_pred=model.predict(X)
   
    v = permute_labels(model.labels_)
    
    c = confusion_matrix(y_true,v)
    r +=1    
    print(r)
    print(c)
    
#sys.stdout.close()       
####################################################################################


# You can plot the Confusion Matrix by using seaborn
# mat = confusion_matrix(digits.target, pred_labels)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=digits.target_names,
#             yticklabels=digits.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label');
