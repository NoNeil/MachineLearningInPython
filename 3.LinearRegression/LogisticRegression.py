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
    return (1.0 / n_samples) * np.dot(x.T, (y - hypothesise(w, x)))

def loss_function(w, x, y):
    n_samples = x.shape[0]
    h = hypothesise(w, x)
    return -(1.0 / n_samples) * np.sum(y.T * np.log(h+1e-6) + (1-y).T * np.log(1-h+1e-6))

def predict(w, x):
    return hypothesise(w, x) > 0.5

def fit_bgd(features, labels, max_iter, alpha):
    n_features = features.shape[1]
    w = np.ones((n_features, 1))
    loss_history = []
    #w = np.mat((np.ones(n_features, 1))
    for i in range(int(max_iter)):
        grad = gradient_decent(w, features, labels)
        loss = loss_function(w, features, labels)
        loss_history.append(loss)
        if i % 1 == 0:
            print("iter={0}, loss={1}".format(i, loss))
            show(df, w)
        w = w + alpha * grad

        pass
    return w, loss_history

def show(df, w):
    # Set style of scatterplot
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    # Create scatterplot of dataframe
    sns.lmplot('x1', # Horizontal axis
               'x2', # Vertical axis
               data=df, # Data source
               fit_reg=False, # Don't fix a regression line
               hue="y", # Set color
               scatter_kws={"marker": "D", # Set marker style
                            "s": 100}) # S marker size

    # Set title
    plt.title('Title')
    plt.xlabel('x1')
    plt.ylabel('x2')

    x = np.linspace(0, 60, 100).reshape(100, 1)
    plt.plot(x, -(w[2] + w[0]*x)/w[1], 'r-')

    plt.interactive(False)
    plt.show(block=True)


#column_names = ['idx', 'x', 'y', '']
#df = pd.read_csv("http://www.stat.ufl.edu/~winner/data/challenger.dat", names=columns)
df_x = pd.read_csv("/Users/Neil/Documents/Code/Github/MachineLearningInPython/3.LinearRegression/data/ex4x.dat", header=None, delim_whitespace=True)
df_y = pd.read_csv("/Users/Neil/Documents/Code/Github/MachineLearningInPython/3.LinearRegression/data/ex4y.dat", delim_whitespace=True)
df = pd.concat([df_x, df_y], axis=1)
df.columns=['x1', 'x2','y']


data = df.as_matrix()
X = data[:-1, 0: -1]
X = np.hstack((X, np.ones((X.shape[0], 1))))
y = np.reshape(data[:-1, -1], [X.shape[0], 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

w, loss_history = fit_bgd(X_train, y_train, 10, 0.001)
print(w)

y_pred = predict(w, X_test)
print(y_pred.T)
print(y_test.T)
print('Accuracy rate: {:d}/{:d} = {:.6f}'.format(np.sum(y_pred == y_test),
              y_test.shape[0],
              np.sum(y_pred == y_test).astype('float') / y_test.shape[0]))


