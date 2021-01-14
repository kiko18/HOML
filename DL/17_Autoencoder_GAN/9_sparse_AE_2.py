# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 01:48:45 2021

@author: BT
"""

'''
Features extraction:
-------------------
In the previous script, we saw how to sparsify the coding layer via l1 regularization.

Another approach, which often yields better results, is to measure the actual sparsity
of the coding layer at each training iteration, and penalize the model when the measured
sparsity differs from a target sparsity. 

We measure the actuel sparsity by computing the average activation of each neuron 
in the coding layer, over the whole training batch. The batch size must not be too small, 
or else the mean will not be accurate.

Once we have the mean activation per neuron, we want to penalize the neurons that
are too active, or not active enough, by adding a sparsity loss to the cost function. 

For example, if we measure that a neuron has an average activation of 0.3, but the target
sparsity is 0.1, it must be penalized to activate less. One approach (to compute the 
sparisty loss to be added to the cost function) could be simply adding the squared error 
(0.3 – 0.1)^2 to the cost function, but in practice a better approach is to use the 
Kullback–Leibler (KL) divergence (briefly discussed in Chapter 4), which has much stronger 
gradients than the mean squared error, as you can see in Figure 17-10.

Once we have computed the sparsity loss for each neuron in the coding layer, we sum
up these losses and add the result to the cost function. 

In order to control the relative importance of the sparsity loss and the reconstruction loss, 
we can multiply the sparsity loss by a sparsity weight hyperparameter. 
If this weight is too high, the model will stick closely to the target sparsity, 
but it may not reconstruct the inputs properly, making the model useless. 
Conversely, if it is too low, the model will mostly ignore the sparsity objective and will not 
learn any interesting features.
'''

'''
Given two discrete probability distributions P and Q, the KL divergence between
these distributions, noted DKL(P ∥ Q), can be computed as sum( P(i) log[P(i)/Q(i)] )

In our case, we want to measure the divergence between the target probability p that a
neuron in the coding layer will activate and the actual probability q (i.e., the mean
activation over the training batch). So the KL divergence simplifies to: 
    DKL(p∥q) = p log(p/q) + 1−p log(1−p/1−q)
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

tf.random.set_seed(42)
np.random.seed(42)

p = 0.1
q = np.linspace(0.001, 0.999, 500)
kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mse = (p - q)**2
mae = np.abs(p - q)
plt.plot([p, p], [0, 0.3], "k:")
plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
plt.plot(q, kl_div, "b-", label="KL divergence")
plt.plot(q, mae, "g--", label=r"MAE ($\ell_1$)")
plt.plot(q, mse, "r--", linewidth=1, label=r"MSE ($\ell_2$)")
plt.legend(loc="upper left", fontsize=14)
plt.xlabel("Actual sparsity")
plt.ylabel("Cost", rotation=0)
plt.axis([0, 1, 0, 0.95])


# We now have all we need to implement a sparse autoencoder based on the KL divergence.
# First, let’s create a custom regularizer to apply KL divergence regularization:

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))
    

# Dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Now we can build the sparse autoencoder, using the KLDivergenceRegularizer for
#the coding layer’s activations:
kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)
])

sparse_kl_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[300]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])

sparse_kl_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
              metrics=[utils.rounded_accuracy])

history = sparse_kl_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))    

utils.show_reconstructions(sparse_kl_ae, X_valid)


utils.plot_activations_histogram(sparse_kl_encoder, X_valid)
plt.show()

'''
After training this sparse autoencoder on Fashion MNIST, the activations of the neurons
in the coding layer are mostly close to 0 (about 70% of all activations are lower
than 0.1), and all neurons have a mean activation around 0.1 (about 90% of all neurons
have a mean activation between 0.1 and 0.2)
'''
