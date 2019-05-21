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