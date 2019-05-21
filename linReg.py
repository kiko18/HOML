#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:08:07 2019

@author: basil
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

#create 2 nodes to hold the data and target
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")


#creata node that will compute theta
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(np.linalg.inv(tf.matmul(XT,X)), XT), y)

#evaluate theta
print(theta.eval)

'''
The main advantage of running this code versus computing the normal equation directely using Numpy is that tensorflow
will automatically run this on you GPU card if you have one ( assuming you installed it with GPU support)
'''


'''
Let use Batch gradient descent instead of normal equation
(When using GD remember that it is important to first normalize the input vectors, or else training may be much slower.
This an be done using Tensorflow, numpy, scikitLearn's standardScaler or any other solution you prefer.)
'''
from sklearn.preprocessing import StandardScaler
scaled_data = StandardScaler().fit_transform(housing.data)
#Verify that the mean of each feature (column) is 0
print(scaled_data.mean(axis = 0))
#Verify that the std of each feature (column) is 1
print(scaled_data.std(axis = 0))