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
#split the data
#X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.33, random_state=42)

'''
Normal Equation
---------------
'''
#initialize feature and response matrix
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

#add theta computation to the graph
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)
loss = tf.reduce_mean(tf.squared_difference(tf.matmul(X, theta), y))

with tf.Session() as sess:
    theta_value, loss_value = sess.run([theta, loss])
    #theta_value = theta.eval()
    #loss_value = loss.eval()
    
print("\n theta normal equation= \n", theta_value)
print(loss_value)

'''
The main advantage of running this code versus computing the normal equation directely using Numpy is that tf
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
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

#Initialise Feature and response matrix
#X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
#y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X") #n+1 because of the bias
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#initialise theta with uniform random value between -1 and 1
theta = tf.Variable(tf.random.uniform([n+1, 1], -1.0, 1.0), name = "theta")
#compute prediction
y_pred = tf.matmul(X, theta, name="predictions")
#compute mean square error (cost)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
#compute the gradient of the cost wrt theta ((2/m) * X^T * (y_pred - y)
#gradients = 2/m * tf.matmul(tf.transpose(X), error)
#tf also has a mechanism to automatically compute the gradients
#gradients = tf.gradients(mse, [theta])[0] #compute gradient of mse wrt theta
#update theta
#training_op = tf.assign(theta, theta - learning_rate * gradients)  

#instead of computing the gradient and updating theta ouself, we can use tf optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
#saver = tf.train.Saver({"weights": theta}) if you only want to save theta
'''
Execution phase
'''


def fetch_batch(epoch, batch_index, batch_size):
    #load the data from disk
    X_batch = scaled_housing_data_plus_bias[batch_size*batch_index : batch_size*(batch_index+1) - 1,:]
    y_batch = housing.target.reshape(-1,1)[batch_size*batch_index : batch_size*(batch_index+1) - 1,:]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        #if(epoch % 100 == 0):   #for each 100 epoch
            #print("Epoch", epoch, "MSE=", mse.eval()) #you need to feed x and y
    #theta does not depend on X and y so we don't need to feed them here 
    best_theta = theta.eval()
    save_path = saver.save(sess, "C:/Users/BT\Documents/others/tf_models/my_linReg_model.ckpt")

print('\n best_theta = \n', best_theta)


tf.reset_default_graph()