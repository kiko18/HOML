# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 01:02:18 2020

@author: BT
"""

'''
Implementing a ResNet-34 CNN Using Keras
----------------------------------------
Most CNN architectures are fairly straightforward to implement.
(although generally you would load a pretrained network instead, as we will see). 
To illustrate the process, let’s implement a ResNet-34 from scratch using Keras. 
First, we create a ResidualUnit layer, then we build the resnet using a sequential model.

It is quite amazing that in less than 40 lines of code, we can build the model that won
the ILSVRC 2015 challenge! It demonstrates both the elegance of the ResNet model,
and the expressiveness of the Keras API. Implementing the other CNN architectures
is not much harder. However, Keras comes with several of these architectures built in,
so why not use them instead?

https://neurohive.io/en/popular-networks/resnet/
'''
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial

def plotTrainingResult(history):
    plt.plot(history.epoch, history.history['loss'], label="train loss")
    plt.plot(history.epoch, history.history['accuracy'], label="train accuracy")
    plt.plot(history.epoch, history.history['val_loss'], label="validation loss")
    plt.plot(history.epoch, history.history['val_accuracy'], label="validation accuracy")
    plt.title('training + validation loss and accuracy')
    plt.legend()
    plt.show()
    
#data
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()    
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]    
    
    
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
inputs = tf.keras.Input(shape=(32, 32, 3))
x = DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[32, 32, 3])(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

prev_filters = 64
# the first 3 Residual unit (RU) has 64 filterd
# then the next 4 RU has 128 filters, and so on
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    # we set the stride to 1 if the number of filters is the same as in the previous RU
    strides = 1 if filters == prev_filters else 2
    x = ResidualUnit(filters, strides=strides)(x)   #add the residual unit
    prev_filters = filters                              #update the previous filterss

x = tf.keras.layers.GlobalAvgPool2D()(x)
x = tf.keras.layers.Flatten()(x)
model_output = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, model_output, name="reset-34")
model.summary()

# compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

plotTrainingResult(history)