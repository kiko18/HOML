# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 01:00:48 2020

@author: BT
"""


'''
CNN Architectures
-----------------
A common mistake is to use convolution kernels that are too large.
For example, instead of using a convolutional layer with a 5 × 5
kernel, it is generally preferable to stack two layers with 3 × 3 kernels:
it will use less parameters and require less computations, and
it will usually perform better. One exception to this recommendation
is for the first convolutional layer: it can typically have a large
kernel (e.g., 5 × 5), usually with stride of 2 or more: this will reduce
the spatial dimension of the image without losing too much information,
and since the input image only has 3 channels in general, it
will not be too costly.

Typically, the number of filters grows as we climb up the CNN towards the output
layer (in the example above it is initially 64, then 128, then 256): 
it makes sense for it to grow, since the number of low level features is 
often fairly low (e.g., small circles, horizontal lines, etc.), but there are many 
different ways to combine them into higher level features. 
It is a common practice to double the number of filters after each pooling
layer: since a pooling layer divides each spatial dimension by a factor of 2, we
can afford doubling the number of feature maps in the next layer, without fear of
exploding the number of parameters, memory usage, or computational load.
'''

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plotTrainingResult(history):
    plt.plot(history.epoch, history.history['loss'], label="train loss")
    plt.plot(history.epoch, history.history['accuracy'], label="train accuracy")
    plt.plot(history.epoch, history.history['val_loss'], label="validation loss")
    plt.plot(history.epoch, history.history['val_accuracy'], label="validation accuracy")
    plt.title('training + validation loss and accuracy')
    plt.legend()
    plt.show()
    
# Example of CNN to tacckle fashion Mnist
# load and preprocess data
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#build CNN
from functools import partial

#partial() defines a thin wrapper around the Conv2D class, called DefaultConv2D: 
#it simply avoids having to repeat the same hyperparameter values over and over again.
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = tf.keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

# compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

plotTrainingResult(history)