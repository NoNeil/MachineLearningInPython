# -*- coding: utf-8 -*-
"""
Created on 2017/8/26
@author: NoNeil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import operator


# load data set
def load_data():
    # data website: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    '''
    data samples with shape of (150, 5):
    sepal_length  sepal_width  petal_length  petal_width        class
    0           5.1          3.5           1.4          0.2  Iris-setosa
    1           4.9          3.0           1.4          0.2  Iris-setosa
    2           4.7          3.2           1.3          0.2  Iris-setosa
    3           4.6          3.1           1.5          0.2  Iris-setosa
    4           5.0          3.6           1.4          0.2  Iris-setosa
    '''
    # define column names
    names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # loading training data
    df = pd.read_csv('./data/iris.data', header=None, names=names)

    # convert category to number
    df['class'] = df['class'].astype('category').cat.codes

    # split features and labels
    X = df.values[:, 1:4]
    y = df.values[:, -1]

    return train_test_split(X, y, test_size=0.3)


# calculate Euclidean Distance
def calc_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# predict with knn
def knn(X_train, y_train, X_test, k):
    y_predict = []
    for X_i_test in X_test:
        k_neighbors = []               # ASC order
        for X_i_train, y_i_train in zip(X_train, y_train):
            dist = calc_euclidean_distance(X_i_test, X_i_train)
            if len(k_neighbors) < k:
                k_neighbors.append((dist, y_i_train))
            else:
                if dist < k_neighbors[-1][0]:
                    k_neighbors[-1] = (dist, y_i_train)
            k_neighbors.sort(key=operator.itemgetter(0))

        # labels of k nearest neighbors
        k_neighbors_label = [neighbor[1] for neighbor in k_neighbors]
        # label of test sample
        y_predict.append(max(k_neighbors_label, key=k_neighbors_label.count))

    return np.array(y_predict)


print('Test for calc_euclidean_distance:', calc_euclidean_distance(np.array([0, 0]), np.array([3, 4])))

# load data
X_train, X_test, y_train, y_test = load_data()
print('Size of X_train:', X_train.shape)
print('Size of X_test:', X_test.shape)

# predict with knn method
y_predict = knn(X_train, y_train, X_test, 7)

# accuracy rate
print('Accuracy rate:', np.sum(y_predict == y_test).astype('float') / y_test.shape[0])
