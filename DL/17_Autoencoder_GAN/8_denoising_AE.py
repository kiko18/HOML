# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:12:56 2021

@author: BT
"""

'''
So far we have seen various kinds of autoencoder (basic, stacked, convolutional, recurrent), 
and we have looked at how to train them (either in one shot or layer by layer). 
We also looked at a couple applications: data visualization and unsupervised pretraining.

Up to now, in order to force the autoencoder to learn interesting features, we have
limited the size of the coding layer, making it undercomplete. There are actually
many other kinds of constraints that can be used, including ones that allow the coding
layer to be just as large as the inputs, or even larger, resulting in an overcomplete
autoencoder. Let’s look at some of those approaches now.
'''

'''
Denoising Autoencoders
----------------------
Another way to force the autoencoder to learn useful features is to add noise to its
inputs, training it to recover the original, noise-free inputs. 
This idea has been around since the 1980s (e.g., it is mentioned in Yann LeCun’s 1987 
master’s thesis). In a 2008 paper,5 Pascal Vincent et al. showed that autoencoders 
could also be used for feature extraction. In a 2010 paper they introduced 
stacked denoising autoencoders.
The noise can be pure Gaussian noise added to the inputs, or it can be randomly
switched-off inputs, just like in dropout.
'''

# The implementation is straightforward: it is a regular stacked autoencoder with an
# additional Dropout layer applied to the encoder’s inputs (or you could use a 
# GaussianNoise layer instead). Recall that the Dropout layer is only active during 
# training (and so is the GaussianNoise layer).

import tensorflow as tf
from tensorflow import keras
import numpy as np
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

# Encoder
denoising_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.5),      #(a)  #noisy images (with half the pixels turned off),
    #keras.layers.GaussianNoise(0.2), #(1)              
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])

# Decoder
denoising_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# AE
denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])

denoising_ae.compile(loss="binary_crossentropy", 
                     optimizer=keras.optimizers.SGD(lr=1.0),
                     metrics=[utils.rounded_accuracy])

history = denoising_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))

#noise = keras.layers.GaussianNoise(0.2) #(2)
noise = keras.layers.Dropout(0.5)  #(b)  #turn half of the pixel off
utils.show_reconstructions(denoising_ae, noise(X_valid, training=True))
plt.show()

'''
The figure shows a few noisy images on top (with half the pixels turned off), and the images 
reconstructed by the dropout-based denoising autoencoder (bottom).

Notice how the autoencoder guesses details that are actually not in the (validation )input, 
such as the top of the white shirt (bottom row, fourth image). 

As you can see, not only can denoising autoencoders be used for data visualization or 
unsupervised pretraining, like the other autoencoders we’ve discussed so far, 
but they can also be used quite simply and efficiently to remove noise from images.
'''

