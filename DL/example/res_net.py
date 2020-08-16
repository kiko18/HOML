# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:33:33 2020

@author: BT
"""

from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
#import tensorflow_datasets as tfds
import numpy as np


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

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1: #skip layers are only needed if the stride is greater than 1
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:  #the input goes through the main layers
            Z = layer(Z)
        skip_Z = inputs                 #the input also goes through the skip connection (if any)
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)  #we add both output and apply the activation fct

# Now, we can build the resnet using a sequential model
# we can treat each residual unit as a single layer
model = tf.keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[28, 28, 1]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
# the first 3 Residual unit (RU) has 64 filterd
# then the next 4 RU has 128 filters, and so on
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    # we set the stride to 1 if the number of filters is the same as in the previous RU
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))   #add the residual unit
    prev_filters = filters                              #update the previous filterss
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))


# compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

plotTrainingResult(history)