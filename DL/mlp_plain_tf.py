# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:59:54 2019

@author: BT
"""
'''
Using the tf estimater is good, but if you want more control over the architecture
of the network, you may prefer to use TF's lower-level Python API.
We will construct the same network as in mlp_estimator.py 
'''

import tensorflow as tf
import numpy as np

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


'''
we use placeholder node to represent the training data and target.
note that X and y here will be use for bach of training data and we don't
know yet how much instance will a batch containt.. 
So the shape of X is (None, n_feature) and the shape of y is None
'''

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")



'''
Create one layer at a time
params:
    -inputs
    -number of neuron
    -activation function
    name of the layer
'''
def neuron_layer(X, n_neurons, name, activation=None):
    #create a name scope with the name of the layer. It will containt all the 
    #computation nodes for this neuron layer. This is optional but the graph will
    #look much nicer in TensorBoard if its node are well organized.
    with tf.name_scope(name): 
        #get the number of input/feature
        n_inputs = int(X.get_shape()[1])
        #initialise the weights W using a truncated normal distribution, with
        #a standard deviation of 2/sqrt(n_inputs + n_neurons)
        #Using this specific standart deviation help the algo converge much faster.
        #W will be a 2D tensor containing all the connection weights 
        #between each input and each neuron, hence of shape (n_inputs, n_neurons).
        #It is important to initialize connections weight randomly for each hidden
        #layers to avoid any symmetries that GD algo would be unable to break
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        #initialize the bias variable with zero
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        #compute the weighted sum of the input and the bias
        #Note that ading 1D vec b to 2D matrix XW, add b to each column of XW, this is 
        #called broadcasting
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


'''
create the DNN
'''
# once again we use a name scoope for clarity
with tf.name_scope("dnn"):
    # the first Hidden layer takes X as its input
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    # the second hidden layer takes the 1st HL as it input
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    # the output layer takes the last HL as its input
    # here as often, logits define the output of the network before going through the 
    #softmax activation function. We don't compute the softmax here because we will
    #incorporate it in the loss function
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
    
    
    
'''
We define the loss function that we will use to train our DNN.
We will use the cross entropy. It penalize models that estimate a low probability 
for the target class. Tf provides several function to compute cross entropy.
We will use sparse_softmax_cross_entropy_with_logits()
it compute the cross entropy based on the "logits" (i.e, the output of the network before
going through the softmax activation function), 
and it expects labels in the form of integers ranging from 0 to n_classes-1 (in our case 0 to 9)
This give us a 1D containing the cross entropy for each instance.
we can then use tf.reduce_mean() to compute the mean cross entropy over all instances.
'''    

with tf.name_scope("loss"):
    #sparse_softmax_cross_entropy_with_logits is equivalent to applying softmax activation
    #function and then computing the cross entropy, but it is more efficient
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
    
    
    
    