# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:47:39 2016

@author: XuXiaoma
"""

import numpy as np

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
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.tolerance = tolerance
        self.labels = y.reshape(y.size, 1)
        #create weights equal to zero with an intercept coefficent at index 0
        self.theta = np.zeros((self.n+1, 1))
        #Add Intercept Data Point of 1 to each row
        self.features = np.ones((self.m, self.n+1))
        self.features[:, :self.n] = X
        self.shuffled_features = self.features
        self.shuffled_labels = self.labels
        self.loss_history = []
        print "initialize"
        
    def hypothesise(self, features):
        return np.dot(features, self.theta)
        
    def gradient(self):
        return (1.0 / self.m) * np.dot(np.transpose(self.features), self.hypothesise(self.features) - self.labels)
        
    def loss_function(self):
        return (0.5 / self.m) * np.sum(np.power(self.hypothesise(self.features) - self.labels, 2))
        
    def gradient_decent(self, alpha=1e-2, max_iter=1e3):        
        for i in range(int(max_iter)):
            pre_loss = self.loss_function()            
            grad = self.gradient()
            self.theta = self.theta - alpha * grad
            self.loss_history.append(pre_loss)
            curr_loss = self.loss_function()
            if(np.abs(pre_loss - curr_loss) < self.tolerance):
                break
            
            