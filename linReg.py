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
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,y)), XT), y)

#evaluate theta
print(theta.eval)