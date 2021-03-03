# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:52:50 2021

@author: BT
"""
import os
import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils


'''
It is common to tie the weights of the encoder and the decoder, by simply using the transpose 
of the encoder's weights as the decoder weights. For this, we need to use a custom layer.

This halves the number of weights in the model, speeding up training and limiting
the risk of overfitting.
'''

'''
This custom layer acts like a regular Dense layer, but it uses another Dense layer’s
weights, transposed (setting transpose_b=True is equivalent to transposing the second
argument, but it’s more efficient as it performs the transposition on the fly within
the matmul() operation). However, it uses its own bias vector.
'''
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
        
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)
        
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True) #transposing the second argument
        return self.activation(z + self.biases)


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

#
dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")

# We can now build a new stacked autoencoder, much like the previous one, 
# but with the decoder’s Dense layers tied to the encoder’s Dense layers
# encoder
tied_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    dense_1,
    dense_2
])

# decoder
tied_decoder = keras.models.Sequential([
    DenseTranspose(dense_2, activation="selu"),
    DenseTranspose(dense_1, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# AE
tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])

# compile and train AE
tied_ae.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1.5), metrics=[utils.rounded_accuracy])

history = tied_ae.fit(X_train, X_train, epochs=10,
                      validation_data=(X_valid, X_valid))

# This model achieves a very slightly lower reconstruction error than the previous
#model, with almost half the number of parameters.
utils.show_reconstructions(tied_ae, X_valid)
plt.show()

