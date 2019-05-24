#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:08:07 2019

@author: basil
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X_train, X_test, y_train, y_test = train_test_split(
        housing_data_plus_bias, housing.target, test_size=0.33, random_state=42)

'''
Normal Equation
---------------
'''
#initialize feature and response matrix
X = tf.constant(X_train, dtype=tf.float32, name="X")
y = tf.constant(y_train.reshape(-1,1), dtype=tf.float32, name="y")

#compute theta
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(np.linalg.inv(tf.matmul(XT,X)), XT), y)

#evaluate theta
print(theta.eval)

#make some prediction
print(mean_squared_error(y, tf.matmul(X, theta)))
print(mean_squared_error(y_test, tf.matmul(tf.constant(X_test, dtype=tf.float32), theta)))

'''
The main advantage of running this code versus computing the normal equation directely using Numpy is that tensorflow
will automatically run this on you GPU card if you have one ( assuming you installed it with GPU support)
'''


'''
Gradient Descent
-----------------
Let use Batch gradient descent instead of normal equation
(When using GD remember that it is important to first normalize the input vectors, or else training may be much slower.
This an be done using Tensorflow, numpy, scikitLearn's standardScaler or any other solution you prefer.)
'''
from sklearn.preprocessing import StandardScaler
scaled_housing_data_plus_bias = StandardScaler().fit_transform(housing_data_plus_bias)
#Verify that the mean of each feature (column) is 0
print(scaled_housing_data_plus_bias.mean(axis = 0))
#Verify that the std of each feature (column) is 1
print(scaled_housing_data_plus_bias.std(axis = 0))

n_epochs = 1000
learning_rate = 0.01

#Initialise Feature and response matrix
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
#initialise theta with uniform random value between -1 and 1
theta = tf.Variable(tf.random.uniform([n+1, 1], -1.0, 1.0), name = "theta")

mse = tf.Variable(tf.zeros([10]))


@tf.function
def train():
    #compute prediction
    y_pred = tf.matmul(X, theta, name="predictions")
    #compute mean square error (cost)
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    #compute the gradient of the cost wrt theta
    #gradients = 2/m * tf.matmul(tf.transpose(X), error)
    gradients = tf.gradients(mse, [theta])[0]
    #update theta
    theta.assign(theta - learning_rate * gradients)  
    return mse


@tf.function
def train2():
    #compute prediction
    y_pred = tf.matmul(X, theta, name="predictions")
    #compute mean square error (cost)
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    #compute the gradient of the cost wrt theta
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer.minimize(mse)
    return mse
    
    
#gradient descent
for epoch in range(n_epochs):
    mse = train()
    if(epoch % 100 == 0):   #for each 100 epoch
        print("Epoch", epoch, "MSE=", mse.eval)

print(mean_squared_error(y, tf.matmul(X, theta)))