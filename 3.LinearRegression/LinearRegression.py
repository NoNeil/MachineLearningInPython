# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:47:39 2016

@author: NoNeil
"""

import numpy as np
import math
import sys


class LinearRegression:

    def __init__(self, X, y, tolerance=1e-5):
        """Initializes Class for Linear Regression

        Parameters
        ----------
        X : ndarray(n-rows,m-features)
            Numerical training data.

        y: ndarray(n-rows,)
            Interger training labels.

        tolerance : float (default 1e-5)
            Stopping threshold difference in the loglikelihood between iterations.

        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.tolerance = tolerance
        self.labels = y.reshape(y.size, 1)
        # create weights equal to zero with an intercept coefficent at index 0
        self.theta = np.zeros((self.n_features + 1, 1))

        # Add Intercept Data Point of 1 to each row
        self.features = np.ones((self.n_samples, self.n_features + 1))
        self.features[:, :self.n_features] = X

        self.loss_history = []
        print("initialize")

    # y = w * x
    def hypothesise(self, features):
        return np.dot(features, self.theta)

    # batch gradient
    # grad = (1 / n_samples) * x.T * (y' - w*x)
    def batch_gradient(self):
        return (1.0 / self.n_samples) * np.dot(np.transpose(self.features),
                                               self.labels - self.hypothesise(self.features))

    # loss = (1 / 2) * (y' - y)^2
    def loss_function(self):
        return (0.5 / self.n_samples) * \
            np.sum(np.power(self.hypothesise(self.features) - self.labels, 2))

    # Batch Gradient Descent
    def batch_gradient_decent(self, alpha=1e-2, max_iter=1e3):
        for i in range(int(max_iter)):
            grad = self.batch_gradient()
            self.theta += alpha * grad      # update theta

            loss = self.loss_function()
            if len(self.loss_history) > 0 and np.abs(
                    self.loss_history[-1] - loss) < self.tolerance:
                break
            self.loss_history.append(loss)
        pass

    # stochastic gradient
    # grad = x_i * (y' - w*x)
    def stochastic_gradient(self, idx):
        x_i = self.features[idx, :].reshape(1, self.n_features + 1)
        y_i = self.labels[idx, :]
        return np.dot(x_i.T, y_i - self.hypothesise(x_i))

    # Stochastic Gradient Descent
    def stochastic_gradient_descent(self, alpha=1e-2, max_iter=1e3):
        for i in range(int(max_iter)):
            # for each samples
            for j in range(self.n_samples):
                grad = self.stochastic_gradient(j)
                self.theta += alpha * grad      # update theta

            loss = self.loss_function()
            if len(self.loss_history) > 0 and np.abs(
                    self.loss_history[-1] - loss) < self.tolerance:
                break
            self.loss_history.append(loss)
        pass
