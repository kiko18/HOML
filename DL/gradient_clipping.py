# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 06:43:39 2019

@author: BT
"""


'''
Beside Batch normalization another popular technique to lessen the exploding gradient problem is to simply clip gradients 
during backpropagation so that they never exceed some threshold (this is mostly used for recurrent neural networks).
This technique is called gradient clipping. In general people prefer batch Normalization, but it's still useful to know
about Gradient Clipping.

Implementation
--------------
In tensorflow, the optimizer's minimize() function takes care of both computing the gradients and appplying them,
so we must instead: 
    - call the optimizer's compute_gradient() method first, 
    - then create an operator to clip the gradient using the clip_by_value() function,
    - finally create an operation to apply the clipped gradients using the optimizer's apply_gradients() method
'''

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10


learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# A hyper params you can tune 
threshold = 1.0

# Optimizer that implements the gradient descent algorithm 
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# compute the gradient
grads_and_vars = optimizer.compute_gradients(loss)
# clip the gradient between -1.0 ans 1.0
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
# operator that apply the clipped gradient. We will then run this operator at every traning step as usual.
# It will compute the gradient, clip them between -1.0 and .0 and apply them.
training_op = optimizer.apply_gradients(capped_gvs)



'''
The rest is the same as usual
'''

# Define the evaluation operation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# create a variable initializer as well as a saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# load the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]



def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
     
        
n_epochs = 20
batch_size = 200
modelParamsDir = 'C:/Users/BT/Documents/others/tf/tf_boards/params/mlp_plain_tf.ckpt'


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, modelParamsDir)














