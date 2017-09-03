# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:50:14 2016

@author: NoNeil
"""

from LinearRegression import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def show_result(lr, title):
    fig, axarr = plt.subplots(2, 1)
    fig.suptitle(title, fontsize=16)

    # Draw line
    num_samples = 1000
    x = lr.features[:, :-1]
    min_val_x, max_val_x = np.floor(np.amin(x)), np.ceil(np.amax(x))
    x_continue = np.linspace(min_val_x, max_val_x, num_samples).reshape(num_samples, 1)
    features = np.hstack((x_continue, np.ones((num_samples, 1))))
    y_continue = lr.hypothesise(features)
    axarr[0].set_title("Draw line.")
    axarr[0].plot(lr.features[:, :-1], lr.labels, 'bo')
    axarr[0].plot(x_continue, y_continue, 'r-')
    axarr[0].axis([min_val_x, max_val_x, np.amin(y_continue), np.amax(y_continue)])

    # History of the Cost
    num_iters = len(lr.loss_history)
    axarr[1].set_title("History of the Cost")
    axarr[1].plot(np.linspace(1, num_iters, num_iters), lr.loss_history, 'b-')
    axarr[1].axis([0, num_iters, 0, np.max(lr.loss_history)])

    plt.interactive(False)
    plt.show(block=True)
    plt.show()

# Test for Linear Regression
# load data
data = pd.read_csv("./lsd.dat", header=None, sep=r"\s+")
data = data.as_matrix()
X = data[:, 0: -1]
y = data[:, -1]


# invoke batch gradient decent
lr = LinearRegression(X, y, tolerance=1e-4)
lr.batch_gradient_decent(0.05, 1e5)
print('LinearRegression, theta of BGD: ', lr.theta.T, ', num of iterations: ', len(lr.loss_history))
show_result(lr, "Method: batch_gradient_decent")


# invoke stochastic gradient decent
lr = LinearRegression(X, y, tolerance=1e-4)
lr.stochastic_gradient_descent(0.03, 1e3)
print('LinearRegression, theta of SGB: ', lr.theta.T, ', num of iterations: ', len(lr.loss_history))
show_result(lr, "Method: stochastic_gradient_descent")


# invoke general newton method
lr = LinearRegression(X, y, tolerance=1e-4)
lr.newton_general()
print('LinearRegression, theta of newton_general: ', lr.theta.T, ', num of iterations: ', len(lr.loss_history))
show_result(lr, "Method: newton_general")


# invoke newton with Armijo search method
lr = LinearRegression(X, y, tolerance=1e-4)
lr.newton_armijo()
print('LinearRegression, theta of newton_armijo: ', lr.theta.T, ', num of iterations: ', len(lr.loss_history))
show_result(lr, "Method: newton_armijo")

# local weight linear regression
taus = np.linspace(1, 0, 5, False)
for tau in np.nditer(taus):
    _, y_estimate = lr.fit_local_weight_lr(X, tau)
    plt.title("local weight linear regression with tau=" + str(tau))
    plt.plot(lr.features[:, :-1], lr.labels, 'bo')
    plt.plot(X, y_estimate, 'r-')
    plt.show()

# Test fot Logistic Regression
# load data
data = pd.read_csv("./apple_juice.dat", header=None, sep=r"\s+")
data = data.as_matrix()
X = data[:, 0: -1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create Logistic Regression model
lr = LogisticRegression(X_train, y_train, tolerance=1e-5)
lr.stochastic_gradient_descent(0.03, 1e3)
print('Logistic Regression, theta of SGD: {0}, num of iterations: {1}'.format(lr.theta.T, len(lr.loss_history)))

# predict
X_test_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
y_pred = lr.predict(X_test_bias)
print('Accuracy rate: {:d}/{:d} = {:.6f}'
      .format(np.sum(y_pred.T == y_test),
              y_test.shape[0],
              np.sum(y_pred.T == y_test).astype('float') / y_test.shape[0]))
pass

