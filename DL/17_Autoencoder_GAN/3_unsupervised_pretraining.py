# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 23:19:49 2021

@author: BT
"""


'''
As we discussed in Chapter 11, if you are tackling a complex supervised task but you
do not have a lot of labeled training data, one solution is to find a neural network that
performs a similar task and reuse its lower layers. This makes it possible to train a
high-performance model using little training data because your neural network won’t
have to learn all the low-level features; it will just reuse the feature detectors learned
by the existing network.

Similarly, if you have a large dataset but most of it is unlabeled, you can first train a
stacked autoencoder using all the data, then reuse the lower layers to create a neural
network for your actual task and train it using the labeled data. For example,
Figure 17-6 shows how to use a stacked autoencoder to perform unsupervised pretraining
for a classification neural network. When training the classifier, if you really
don’t have much labeled training data, you may want to freeze the pretrained layers
(at least the lower ones).

Having plenty of unlabeled data and little labeled data is common.
Building a large unlabeled dataset is often cheap (e.g., a simple
script can download millions of images off the internet), but labeling
those images (e.g., classifying them as cute or not) can usually
be done reliably only by humans. Labeling instances is timeconsuming
and costly, so it’s normal to have only a few thousand human-labeled instances.

There is nothing special about the implementation: just train an autoencoder using
all the training data (labeled plus unlabeled), then reuse its encoder layers to create a
new neural network (see the exercises at the end of this chapter for an example).
'''

import os
import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

tf.random.set_seed(42)
np.random.seed(42)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_train_small = X_train[:500]
y_train_small = y_train[:500]

# Let's create a small neural network for MNIST classification (without pretraining)
classifier = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(20, activation="selu"),
    keras.layers.Dense(10, activation="softmax")
])
classifier.compile(loss="sparse_categorical_crossentropy", 
                   optimizer=keras.optimizers.SGD(lr=0.02),
                   metrics=["accuracy"])

history = classifier.fit(X_train_small, y_train_small, epochs=20, validation_data=(X_valid, y_valid))
accuracy_without_pretraining = classifier.evaluate(X_test,y_test)[1]

import pandas as pd
pd.DataFrame(history.history).plot()
plt.show()

#-------------------------------------
# consider only X_train_small is labelled. We use unsupervised pretrained.
# We train an AE on the entire dataset (i.e X_train). 
# Then we will reuse the encoder part, which should have learn the core features.
## Train a deep denoising autoencoder on the full training set.
# Encoder
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])

# Decoder
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# Auto-Encoder
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# compile AE
conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                metrics=[utils.rounded_accuracy])
# train AE
history = conv_ae.fit(X_train, X_train, epochs=5,
                      validation_data=(X_valid, X_valid))

#accuracy_ae = conv_ae.evaluate(X_test,X_test)[1]

#-----------------------------------------------------------
# Build a classification DNN, reusing the lower layers of the autoencoder.
# Train it using only 500 images from the training set. 
# Does it perform better with or without pretraining?
conv_encoder_clone = keras.models.clone_model(conv_encoder)

pretrained_clf = keras.models.Sequential([
    conv_encoder_clone,
    keras.layers.Flatten(),
    keras.layers.Dense(20, activation="selu"),
    keras.layers.Dense(10, activation="softmax")
])

## with freezed pretrained wieght
#When training the classifier, if you really don’t have much labeled training data, 
#you may want to freeze the pretrained layers (at least the lower ones).
conv_encoder_clone.trainable = False
pretrained_clf.compile(loss="sparse_categorical_crossentropy",
                       optimizer=keras.optimizers.SGD(lr=0.02),
                       metrics=["accuracy"])

history = pretrained_clf.fit(X_train_small, y_train_small, epochs=30,
                             validation_data=(X_valid, y_valid))

accuracy_with_frozen_pretrained_ae = pretrained_clf.evaluate(X_test,y_test)[1]

## with unfreezed pretrained wieght
conv_encoder_clone.trainable = True
pretrained_clf.compile(loss="sparse_categorical_crossentropy",
                       optimizer=keras.optimizers.SGD(lr=0.02),
                       metrics=["accuracy"])
history = pretrained_clf.fit(X_train_small, y_train_small, epochs=20,
                             validation_data=(X_valid, y_valid))

accuracy_with_unfrozed_pretrained_ae = pretrained_clf.evaluate(X_test,y_test)[1]

print("\n \n")
print('accuracy_without_pretraining         =', round(accuracy_without_pretraining,5))
print('accuracy_with_frozen_pretrained_ae   =', round(accuracy_with_frozen_pretrained_ae,5))
print('accuracy_with_unfrozed_pretrained_ae =', round(accuracy_with_unfrozed_pretrained_ae,5))