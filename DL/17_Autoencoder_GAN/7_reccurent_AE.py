# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:33:13 2021

@author: BT
"""

'''
If you want to build an autoencoder for sequences, such as time series or text (e.g., for
unsupervised learning or dimensionality reduction), then recurrent neural networks
(see Chapter 15) may be better suited than dense networks. Building a recurrent
autoencoder is straightforward: the encoder is typically a sequence-to-vector RNN
which compresses the input sequence down to a single vector. The decoder is a
vector-to-sequence RNN that does the reverse.
'''

import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils


# Load the dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Encoder
recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=[28, 28]),
    keras.layers.LSTM(30)
])

# Decoder
# use a RepeatVector layer as the first layer of the,decoder, 
# to ensure that its input vector gets fed to the decoder at each time step
recurrent_decoder = keras.models.Sequential([
    keras.layers.RepeatVector(28, input_shape=[30]),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
])
recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])
recurrent_ae.compile(loss="binary_crossentropy", 
                     optimizer=keras.optimizers.SGD(0.1),
                     metrics=[utils.rounded_accuracy])


# This recurrent autoencoder can process sequences of any length, with 28 dimensions
# per time step. Conveniently, this means it can process Fashion MNIST images by
# treating each image as a sequence of rows: at each time step, the RNN will process a
# single row of 28 pixels. Obviously, you could use a recurrent autoencoder for any
# kind of sequence
history = recurrent_ae.fit(X_train, X_train, epochs=10, validation_data=(X_valid, X_valid))

utils.show_reconstructions(recurrent_ae, X_valid)
plt.show()