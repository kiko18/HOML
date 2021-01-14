# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:51:42 2021

@author: BT
"""

'''
Features extraction:
-------------------
    
Another kind of constraint that often leads to good feature extraction is sparsity: by
adding an appropriate term to the cost function, the autoencoder is pushed to reduce
the number of active neurons in the coding layer. 

For example, it may be pushed to have on average only 5% significantly active neurons 
in the coding layer. This forces the autoencoder to represent each input as a combination 
of a small number of activations.

As a result, each neuron in the coding layer typically ends up representing a
useful feature (if you could speak only a few words per month, you would probably
try to make them worth listening to).

To sparsify the coding layer, a simple approach is to use the sigmoid activation function 
in the coding layer (to constrain the codings to values between 0 and 1), 
use a large coding layer (e.g., with 300 units), and add some ℓ1 regularization to the 
coding layer’s activations (the decoder is just a regular decoder)
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

tf.random.set_seed(42)
np.random.seed(42)

# Dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# Simple AE
#---------------------------------------
# Let's build a simple stacked autoencoder, so we can compare it to the sparse 
# autoencoders we will build. This time we will use the sigmoid activation function 
# for the coding layer, to ensure that the coding values range from 0 to 1.
# Simple Encoder
simple_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="sigmoid"), #coging layer
])

# Simple Decoder
simple_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# Simple AE
simple_ae = keras.models.Sequential([simple_encoder, simple_decoder])

simple_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.),
                  metrics=[utils.rounded_accuracy])

history = simple_ae.fit(X_train, X_train, epochs=10,
                        validation_data=(X_valid, X_valid))


utils.show_reconstructions(simple_ae, X_valid)
plt.show()
  

# Let's  plot histograms of the activations of the encoding layer. 
# The histogram on the left shows the distribution of all the activations. 
# You can see that values close to 0 or 1 are more frequent overall, which is consistent 
# with the saturating nature of the sigmoid function. 
# The histogram on the right shows the distribution of mean neuron activations: 
# you can see that most neurons have a mean activation close to 0.5. 
# Both histograms tell us that each neuron tends to either fire close to 0 or 1, 
# with about 50% probability each. 
# However, some neurons fire almost all the time (right side of the right histogram).
utils.plot_activations_histogram(simple_encoder, X_valid, height=0.35)
plt.show()


#Sparse AE (let's add l1 regularization to the coding layer)
#-----------------------------------------------------------
# sparse encoder
sparse_l1_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid"),
    keras.layers.ActivityRegularization(l1=1e-3)  #add l1 loss term to the loss fct
    # Alternatively, you could add
    # activity_regularizer=keras.regularizers.l1(1e-3)
    # to the previous layer.
])

# sparse decoder
sparse_l1_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[300]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# sparse AE
sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])

sparse_l1_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                     metrics=[utils.rounded_accuracy])

history = sparse_l1_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))

utils.show_reconstructions(sparse_l1_ae, X_valid)

utils.plot_activations_histogram(sparse_l1_encoder, X_valid, height=1.)
plt.show()

'''
This ActivityRegularization layer just returns its inputs, but as a side effect it adds
a training loss equal to the sum of absolute values of its inputs (this layer only has an
effect during training). 

Equivalently, you could remove the ActivityRegularization layer and set 
activity_regularizer=keras.regularizers.l1(1e-3) in the previous layer. 

This penalty will encourage the neural network to produce codings close to 0,
but since it will also be penalized if it does not reconstruct the inputs correctly, it will
have to output at least a few nonzero values. 

Using the ℓ1 norm rather than the ℓ2 norm will push the neural network to preserve the most 
important codings while eliminating the ones that are not needed for the input image 
(rather than just reducing all codings).
'''