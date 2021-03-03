# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:15:10 2021

@author: BT
"""

'''
Variational Autoencoders (VA)
-----------------------------
VA are quite different from all the autoencoders we have discussed so far, in these
particular ways:
    • They are probabilistic autoencoders, meaning that their outputs are partly determined
        by chance, even after training (as opposed to denoising autoencoders, which use 
        randomness only during training).
    • Most importantly, they are generative autoencoders, meaning that they can generate
        new instances that look like they were sampled from the training set.

Indeed, as their name suggests, variational autoencoders perform variational Bayesian
inference (introduced in Chapter 9), which is an efficient way to perform approximate 
Bayesian inference.

How do their work? 
------------------
In VA, instead of directly producing a coding for a given input, the encoder produces a
mean coding μ and a standard deviation σ. 

The actual coding is then sampled randomly from a Gaussian distribution with mean μ and 
standard deviation σ. 

After that the decoder decodes the sampled coding normally. 

The right part of the diagram shows a training instance going through this autoencoder. 
First, the encoder produces μ and σ, then a coding is sampled randomly (notice that it is 
not exactly located at μ), and finally this coding is decoded; the final output resembles 
the training instance.

As you can see in the diagram, although the inputs may have a very convoluted distribution,
a variational autoencoder tends to produce codings that look as though they
were sampled from a simple Gaussian distribution: during training, the cost function
(discussed next) pushes the codings to gradually migrate within the coding space
(also called the latent space) to end up looking like a cloud of Gaussian points. 
One great consequence is that after training a variational autoencoder, you can very easily
generate a new instance: just sample a random coding from the Gaussian distribution,
decode it, and voilà!

VA Cost Function
----------------
It is composed of two parts. The first is the usual reconstruction loss that pushes the 
autoencoder to reproduce its inputs (we can use cross entropy for this, as discussed earlier). 
The second is the latent loss that pushes the autoencoder to have codings that look as though 
they were sampled from a simple Gaussian distribution: it is the KL divergence between the target 
distribution (i.e., the Gaussian distribution) and the actual distribution of the codings. 
The math is a bit more complex than with the sparse autoencoder, in particular because of the 
Gaussian noise, which limits the amount of information that can be transmitted to the
coding layer (thus pushing the autoencoder to learn useful features). 

Luckily, the equations simplify, so the latent loss can be computed quite simply as:
                K
    L = -1/2 [ sum (1 + log((σ_i)^2) - (σ_i)^2 - (mu_i)^2) ]
               i=1

In this equation, L is the latent loss, K is the codings’ dimensionality, μi and σi are
the mean and standard deviation of the ith component of the codings. The vectors μ
and σ (which contain all the μi and σi) are output by the encoder, as shown in
Figure 17-12 (left).

A common tweak to the variational autoencoder’s architecture is to make the encoder
output γ = log(σ^2) rather than σ. The latent loss can then be computed as:
                K
    L = -1/2 [ sum (1 + γ_i - exp(γ_i) - (mu_i)^2) ]    (EQ 17-4)
               i=1

This approach is more numerically stable and speeds up training.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

K = keras.backend

tf.random.set_seed(42)
np.random.seed(42)

# Dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Let’s start building a variational autoencoder for Fashion MNIST (as shown in Figure 17-12, 
# but using the γ tweak). 
# First, we will need a custom layer to sample the codings, given μ and γ.

# Class that samples a codings vector from the Normal distribution with mean μ and std σ.
class Sampling(keras.layers.Layer):
    # This Sampling layer takes two inputs: mean (μ) and log_var (γ). It uses the function
    # K.random_normal() to sample a random vector (of the same shape as γ) from the
    # Normal distribution, with mean 0 and standard deviation 1. Then it multiplies it by
    # exp(γ / 2) (which is equal to σ, as you can verify), and finally it adds μ and returns the
    # result. This samples a codings vector from the Normal distribution with mean μ and
    # standard deviation σ.
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

# Encoder
# we use the Functional API because the model is not entirely sequential
codings_size = 10

inputs = keras.layers.Input(shape=[28, 28])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
#Note that the Dense layers that output codings_mean (μ) and codings_log_var (γ)
#have the same inputs (i.e., the outputs of the second Dense layer). 
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
# We then pass both codings_mean and codings_log_var to the Sampling layer. 
codings = Sampling()([codings_mean, codings_log_var])
# Finally, the varia tional_encoder model has three outputs (in case you want to inspect 
# the values of codings_mean and codings_log_var. 
# The only output we will use is the last one (codings).
variational_encoder = keras.models.Model(inputs=[inputs], 
                                         outputs=[codings_mean, codings_log_var, codings])

 
# Decoder
decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = keras.layers.Reshape([28, 28])(x)
variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

# For this decoder, we could have used the Sequential API instead of the Functional API, 
# since it is really just a simple stack of layers, virtually identical to many of the
# decoders we have built so far. Finally, let’s build the variational autoencoder model

# Variational Autoencoder (VA) model
_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

# Note that we ignore the first two outputs of the encoder (we only want to feed the
# codings to the decoder). 
# Lastly, we must add the latent loss and the reconstruction loss 

# We first apply Equation 17-4 to compute the latent loss for each instance in the batch
# (we sum over the last axis). Then we compute the mean loss over all the instances in
# the batch, and we divide the result by 784 to ensure it has the appropriate scale compared
# to the reconstruction loss. Indeed, the variational autoencoder’s reconstruction
# loss is supposed to be the sum of the pixel reconstruction errors, but when Keras
# computes the "binary_crossentropy" loss, it computes the mean over all 784 pixels  rather than 
# the sum. So, the reconstruction loss is 784 times smaller than we need it to be. 
# We could define a custom loss to compute the sum rather than the mean, but it
# is simpler to divide the latent loss by 784 (the final loss will be 784 times smaller than
# it should be, but this just means that we should use a larger learning rate).
latent_loss = -0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis=-1)

variational_ae.add_loss(K.mean(latent_loss) / 784.)

# Note that we use the RMSprop optimizer, which works well in this case. 
variational_ae.compile(loss="binary_crossentropy", 
                       optimizer="rmsprop", 
                       metrics=[utils.rounded_accuracy])

# And finally we can train the autoencoder!
history = variational_ae.fit(X_train, X_train, 
                             epochs=25, 
                             batch_size=128,
                             validation_data=(X_valid, X_valid))

utils.show_reconstructions(variational_ae, X_test)
plt.show()


'''
Generate Images
'''
# Let's generate a few random codings, decode them and plot the resulting images:
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()
utils.plot_multiple_images(images, 4)
#save_fig("vae_generated_images_plot", tight_layout=False)


# Now let's perform semantic interpolation between these images
codings_grid = tf.reshape(codings, [1, 3, 4, codings_size])
larger_grid = tf.image.resize(codings_grid, size=[5, 7])
interpolated_codings = tf.reshape(larger_grid, [-1, codings_size])
images = variational_decoder(interpolated_codings).numpy()

plt.figure(figsize=(7, 5))
for index, image in enumerate(images):
    plt.subplot(5, 7, index + 1)
    if index%7%2==0 and index//7%2==0:
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    else:
        plt.axis("off")
    plt.imshow(image, cmap="binary")
#save_fig("semantic_interpolation_plot", tight_layout=False)







