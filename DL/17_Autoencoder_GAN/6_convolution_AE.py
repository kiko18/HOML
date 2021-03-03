# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:48:00 2021

@author: BT
"""

'''
If you are dealing with images, then the autoencoders we have seen so far will not
work well (unless the images are very small). If you want to build an autoencoder 
for images (e.g., for unsupervised pretraining or dimensionality reduction), 
you will need to build a convolutional autoencoder. 

In convolutional AE, the encoder is a regular CNN composed of convolutional layers 
and pooling layers. It typically reduces the spatial dimensionality of the inputs 
(i.e., height and width) while increasing the depth (i.e., the number of feature maps). 

The decoder must do the reverse (upscale the image and reduce its depth back to the 
original dimensions), and for this you can use transpose convolutional layers 
(alternatively, you could combine upsampling layers with convolutional layers).
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import utils

#Let's build a stacked Autoencoder with 3 hidden layers and 1 output layer 
#(i.e., 2 stacked Autoencoders).
tf.random.set_seed(42)
np.random.seed(42)

# Dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# encoder
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])

# decoder
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# AE
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                metrics=[utils.rounded_accuracy])

history = conv_ae.fit(X_train, X_train, epochs=5, validation_data=(X_valid, X_valid))

conv_encoder.summary()
conv_decoder.summary()

utils.show_reconstructions(conv_ae, X_valid, title='')
plt.show()

