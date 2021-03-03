# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:35:45 2021

@author: BT
"""

'''
Training One Autoencoder at a Time
----------------------------------
Rather than training the whole stacked autoencoder in one go like we just did, it is
possible to train one shallow autoencoder at a time, then stack all of them into a single
stacked autoencoder (hence the name).

This technique is not used as much these days, but you may still run into papers that 
talk about “greedy layerwise training,” so it’s good to know what it means.
'''

import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils


tf.random.set_seed(42)
np.random.seed(42)

def train_autoencoder(n_neurons, X_train, X_valid, loss, optimizer,
                      n_epochs=10, output_activation=None, metrics=None):
    n_inputs = X_train.shape[-1]
    encoder = keras.models.Sequential([
        keras.layers.Dense(n_neurons, activation="selu", input_shape=[n_inputs])
    ])
    decoder = keras.models.Sequential([
        keras.layers.Dense(n_inputs, activation=output_activation),
    ])
    autoencoder = keras.models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer, loss, metrics=metrics)
    autoencoder.fit(X_train, X_train, epochs=n_epochs,
                    validation_data=(X_valid, X_valid))
    return encoder, decoder, encoder(X_train), encoder(X_valid)

# Load the dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

K = keras.backend

X_train_flat = K.batch_flatten(X_train) # equivalent to .reshape(-1, 28 * 28)
X_valid_flat = K.batch_flatten(X_valid)

# train the first AE
# During this first phase of training, this first autoencoder learns to reconstruct the
# inputs. Then we encode the whole training set using this first autoencoder, and this
# gives us a new (compressed) training set X_train_enc1, X_valid_enc1 
enc1, dec1, X_train_enc1, X_valid_enc1 = train_autoencoder(100, X_train_flat, X_valid_flat, 
                                                           "binary_crossentropy",
                                                           keras.optimizers.SGD(lr=1.5), 
                                                           output_activation="sigmoid",
                                                           metrics=[utils.rounded_accuracy])

# second AE
# We then train a second autoencoder on this new dataset. 
# This is the second phase of training. 
enc2, dec2, _, _ = train_autoencoder(30, X_train_enc1, X_valid_enc1, "mse", 
                                     keras.optimizers.SGD(lr=0.05),
                                     output_activation="selu")

# Finally, we build a big sandwich using all these autoencoders.
# i.e., we first stack the hidden layers of each autoencoder (the encoder layers), 
# then the output layers in reverse order).
# We could easily train more autoencoders this way, building a very deep stacked autoencoder.
stacked_ae_1_by_1 = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                             enc1, enc2, dec2, dec1,
                                             keras.layers.Reshape([28, 28])
])

utils.show_reconstructions(stacked_ae_1_by_1, X_valid, title='before training')
plt.show()


stacked_ae_1_by_1.compile(loss="binary_crossentropy",
                          optimizer=keras.optimizers.SGD(lr=0.1), 
                          metrics=[utils.rounded_accuracy])

history = stacked_ae_1_by_1.fit(X_train, X_train, epochs=10,
                                validation_data=(X_valid, X_valid))

utils.show_reconstructions(stacked_ae_1_by_1, X_valid, title='after training')
plt.show()

'''
As we discussed earlier, one of the triggers of the current tsunami of interest in Deep
Learning was the discovery in 2006 by Geoffrey Hinton et al. that deep neural networks
can be pretrained in an unsupervised fashion, using this greedy layerwise
approach. They used restricted Boltzmann machines (RBMs; see Appendix E) for this
purpose. 

In 2007 Yoshua Bengio et al. showed that autoencoders worked just as well. 
For several years this was the only efficient way to train deep nets, until many of
the techniques introduced in Chapter 11 made it possible to just train a deep net in
one shot.
'''