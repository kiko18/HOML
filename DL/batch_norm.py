# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:51:15 2019

@author: BT
"""

'''
Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce
the vanishing/exploding gradients problems at the beginning of traning, it doesn't garantee that
they won't come back during traning.
In a 2015 paper, Sergey Ioffe and Christian Szegedy proposed a technique called Batch normalization (BN)
to adress the vanishing/exploding gradient problems, and more generally the problem that the distribution
of each layer's inputs changes during training, as the parameters of the previous layers change (which
they call Internal Covariante Shift problem).

The technique consists of adding an operation in the model just before the activation function of each layer.
Simply zero-centering and normalizing the inputs, then scaling and shifting the result using 2 new parameters
per layer (one for scaling the other for shifting). In others words, this operation lets the model learn 
the optimal scale and mean of the inputs for each layer.

The process is as follow:
1- compute the mean
2- compute the standart deviation
3- compute the zero-centered and normalized inputs
4- scale and shift (offset) the normalized input

So, in total, 4 parameters are learned for each batch-normalized layer: scale, offset, mean and 
standart deviation.
'''

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

from functools import partial


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#The batch-normalization algorithm uses exponential decay to compute the running averages
#which is why it requires the momentum parameter. Given a new value v, the running average v_hat
#is updated through the equation: v_hat <- v_hat * momentum + v * (1-momentum)
batch_norm_momentum = 0.9   #a good momentum value is typically close to one
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

#define a training placehoder that will be set to True during training and to false otherwise 
#to tell the tf.layers.batch_normalization() function wheter it should use the current 
#mini-batch's mean and standard deviation (during training) or the whole set's mean and standard deviation 
#(during testing)
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    #To avoid repeating the same parameters over and over again, we can use Python's partial() function:
    my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)

    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)
    hidden2 = my_dense_layer(bn1_act, n_hidden2, name="hidden2")
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)
    logits_before_bn = my_dense_layer(bn2_act, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

'''
Define a cost function
'''
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

'''
Create a gradient optimizer and tell it to minimize the cost
'''
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

'''
Define the evaluation operation
'''
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
'''
create a variable initializer as well as a saver
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()

'''
load the data
'''
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
        

'''
Since we are using tf.layers.batch_normalization() rather than tf.contrib.layers.batch_norm()
we need to explicitly run the extra update operations needed by batch normalization 
(sess.run([training_op, extra_update_ops],...).
'''        
n_epochs = 20
batch_size = 200

'''
Batch normalization() creates a few operations that must be evaluated at each step during training in 
order to update the mooving averages (recall that these moving averages are needed to evaluate the trainig 
set's mean and standard deviation). These operation (bachnorm add) are automatically added to the UPDATE_OPS collection,
so all we need to do is get the list of operations in that collection and run them at each training iteration.
'''
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

modelParamsDir = 'C:/Users/BT/Documents/others/tf/tf_boards/params/mlp_plain_tf.ckpt'

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            #training_op depend on batch_normalization(), this implies we need to set training placeholder to true
            sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, modelParamsDir)
    
'''
And that's all! In this tiny example with just two layers, it's unlikely that batch normalization will have 
a very positive impact, but for deeper networks it can make a tremendous difference.

One more thing: notice that the list of trainable variables is shorter than the list of all global variables. 
This is because the moving averages are non-trainable variables. 
If you want to reuse a pretrained neural network (see below), you must not forget these non-trainable variables.
'''    
[v.name for v in tf.trainable_variables()]
[v.name for v in tf.global_variables()]
    
'''

SELU

This activation function was proposed in this great paper by Günter Klambauer, Thomas Unterthiner and 
Andreas Mayr, published in June 2017. During training, a neural network composed exclusively of a stack of 
dense layers using the SELU activation function and LeCun initialization will self-normalize: 
the output of each layer will tend to preserve the same mean and variance during training, 
which solves the vanishing/exploding gradients problem. As a result, this activation function outperforms 
the other activation functions very significantly for such neural nets, so you should really try it out. 
Unfortunately, the self-normalizing property of the SELU activation function is easily broken: 
you cannot use ℓ1 or ℓ2 regularization, regular dropout, max-norm, skip connections or other non-sequential 
topologies (so recurrent neural networks won't self-normalize). However, in practice it works quite well 
with sequential CNNs. If you break self-normalization, SELU will not necessarily outperform other activation 
functions.
'''
from scipy.special import erfc

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

# alpha and scale to self normalize with mean 0 and standard deviation 1
# (see equation 14 in the paper):
alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e**2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)

def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)

'''
By default, the SELU hyperparameters (scale and alpha) are tuned in such a way that the mean output of each 
neuron remains close to 0, and the standard deviation remains close to 1 (assuming the inputs are standardized
with mean 0 and standard deviation 1 too). Using this activation function, even a 1,000 layer deep neural 
network preserves roughly mean 0 and standard deviation 1 across all layers, avoiding the exploding/vanishing 
gradients problem:
'''
np.random.seed(42)
Z = np.random.normal(size=(500, 100)) # standardized inputs
for layer in range(1000):
    W = np.random.normal(size=(100, 100), scale=np.sqrt(1 / 100)) # LeCun initialization
    Z = selu(np.dot(Z, W))
    means = np.mean(Z, axis=0).mean()
    stds = np.std(Z, axis=0).mean()
    if layer % 100 == 0:
        print("Layer {}: mean {:.2f}, std deviation {:.2f}".format(layer, means, stds))


