#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:12:09 2019

@author: bt
"""

'''
In ML Training a model means searching for a combination of model parameters that minimizes 
a cost function (over the trainig set). It is a search in the model's parameters space.
The more parameter a model has, the more dimensions this space has, and the harder the search is.
Searching for a needle in a 300-dimensional haystack is much trickier than in 3 dimensions.

There are 2 differents way of training models in ML.
- using a direct "closed-form" equation that directely computes the model parameters 
  that best fit the model to the tranning set (ie. the model paramater that minimize 
  the cost function over the traning set)
- Using an iterative optimization approach called Gradient Descent (GD) that gradually 
  tweaks the model parameters to minimize the cost function over the traning set. 
  Eventually converging to the same set of parameters as the first method.
  
We will first look at linear regression, a model capable of fitting linear data.
Later on, we will look at Polinomial regression, which is a more complex model that can fit
non-linear datasets. Since this model has more parameters than linear Regression, it is
more prone to overfitting the traning data. So we will learn to detect whether or not
this is the case using learning curves and we will look at several regularization techniques 
that can reduce the risk of overfitting the traning set.
'''
import numpy as np
import matplotlib.pyplot as plt

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)   #y = 4+3x+Gaussian_noise -> theta_0 should 4 and theta_ should be 3

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

#add bias
X_b = np.c_[np.ones((100,1)), X]    #add X= = 1 to each instance

'''
Normal Equation
'''

#compute theta (due to noise we can't recover the exact params, but we should be close enaugh)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#make predictions using theta
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new] 
y_predict = X_new_b.dot(theta_best)

#plot the model prediction
plt.plot(X_new, y_predict, "r-", linewidth=2, label="linear_model_predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()

#linear Regression can also be done with scikit-learn

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('intercept: ', lin_reg.intercept_, 'coeficient: ', lin_reg.coef_)
lin_reg.predict(X_new)


'''
Gradient Descent
'''
# Gradient descent is a generic optimization algorithm capable of finding optimal solution to a wide
# range of problem. 
# The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.
# Supose you are lost in the mountains in a dense fog, and you can only feel the slope of the ground below 
# your feet. A good strategy to get to the bottom of the valley quickly is to go downhill in the direction 
# of the steepest slope. 
# This is exactely what GD does:
# it measures the local gradient of the error function with regard the parameter vector THETA
# and it goes in the direction of the descending gradient. Once the gradient is zero, you've
# reached a minimum! 
# Concretely, you start by filling THETA with random values (random initialization).
# Then you improve it gradually, taking one baby step at a time, each step attempting to decrease
# the cost function (eg. MSE), until the algorithm converges to a minimum.






























