# -*- coding: utf-8 -*-
"""
Created on 2017/9/10
@author: NoNeil
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def sig(z):
    return 1.0 / (1 + np.exp(-z))

def hypothesise(w, x):
    return sig(np.dot(x, w))

def gradient_decent(w, x, y):
    n_samples = x.shape[0]
    return np.dot(x.T, (y - hypothesise(w, x)))

def loss_function(w, x, y):
    n_samples = x.shape[0]
    h = hypothesise(w, x)
    return (1.0 / n_samples) * np.sum(y.T * np.log(h+1e-6) + (1-y).T * np.log(1-h+1e-6))

def predict(w, x, add_intercept=False):
    if add_intercept:
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))
    return hypothesise(w, x) > 0.5

def fit_bgd(features, labels, max_iter, alpha, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    n_features = features.shape[1]
    w = np.ones((n_features, 1))
    loss_history = []
    for i in range(max_iter):
        grad = gradient_decent(w, features, labels)
        loss = loss_function(w, features, labels)
        loss_history.append(loss)
        if i % 100 == 0:
            print("iter={0}, loss={1}".format(i, loss))
        w = w + alpha * grad

    return w, loss_history

# Test 1
np.random.seed(12)
num_observations = 500
x1 = np.random.multivariate_normal([0, 0], [[1, 1],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 1],[.75, 1]], num_observations)

simulated_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
simulated_labels = simulated_labels[:,np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(simulated_features, simulated_labels, test_size=0.3)

add_intercept = True
w, loss_history = fit_bgd(X_train, y_train, 500, 1e-2, add_intercept=add_intercept)
print('weights: ', w)
y_pred = predict(w, X_test, add_intercept=add_intercept)
print('Accuracy rate: {:d}/{:d} = {:.6f}'.format(np.sum(y_pred == y_test),
                                                 y_test.shape[0],
                                                 np.sum(y_pred == y_test).astype('float') / y_test.shape[0]))

plt.figure(figsize=(12, 8))
plt.scatter(simulated_features[:, 0],
            simulated_features[:, 1],
            c = simulated_labels,
            alpha = 1.0)

x = np.linspace(-4, 4, 100).reshape(100, 1)
plt.plot(x, -(w[0] + w[1]*x)/w[2], 'r-')
plt.show()


# Test 2
df_x = pd.read_csv("./data/ex4x.dat", header=None, delim_whitespace=True)
df_y = pd.read_csv("./data/ex4y.dat", delim_whitespace=True)
df = pd.concat([df_x, df_y], axis=1)
df.columns=['x1', 'x2','y']

data = df.as_matrix()
X = data[:-1, 0: -1]
y = np.reshape(data[:-1, -1], [X.shape[0], 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

add_intercept = True
w, loss_history = fit_bgd(X_train, y_train, 1000, 1e-3, add_intercept=add_intercept)
print('weights: ', w)
y_pred = predict(w, X_test, add_intercept=add_intercept)
print('Accuracy rate: {:d}/{:d} = {:.6f}'.format(np.sum(y_pred == y_test),
                                                 y_test.shape[0],
                                                 np.sum(y_pred == y_test).astype('float') / y_test.shape[0]))

plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0],
            X[:, 1],
            c = y,
            alpha = 1.0)

x = np.linspace(0, 60, 100).reshape(100, 1)
plt.plot(x, -(w[0] + w[1]*x)/w[2], 'r-')
plt.show()
