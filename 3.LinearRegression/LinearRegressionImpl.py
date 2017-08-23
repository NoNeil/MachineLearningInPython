# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:50:14 2016

@author: XuXiaoma
"""

from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_result(X, y, lr):
    print "Result:"
    X1 = np.linspace(0, 8, 1000)
    features = np.ones((X1.shape[0], 2))
    features[:, 0] = X1
    y1 = lr.hypothesise(features)
    plt.plot(X, y, 'bo')
    plt.plot(X1, y1, 'r-')
    plt.axis([0, 8, 0, 100])
    plt.show()
    
def show_loss_history(loss_history):
    print "History of the Cost:"
    num_iters = len(loss_history)
    plt.plot(np.linspace(1, num_iters, num_iters), loss_history, 'b-')
    plt.axis([-20, num_iters, 0, np.max(loss_history)])
    plt.show()

# load data
data = pd.read_csv("./lsd.dat", header=None, sep=r"\s+")
data = data.as_matrix()
X = data[:, 0: 1]
y = data[:, 1: 2]

# invoke gradient decent
lr = LinearRegression(X, y, tolerance=1e-4)
lr.gradient_decent(0.05, 1e5)
print lr.theta

show_result(X, y, lr)
show_loss_history(lr.loss_history)