# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:45:59 2021

@author: BT
"""

'''
Autoencoder -> dense representation
-----------------------------------
Autoencoders are ANN capable of learning dense representations of the input data, 
called latent representations or codings, without any supervision (i.e.,
the training set is unlabeled).

These codings typically have a much lower dimensionality than the input data, 
making autoencoders useful for dimensionality reduction, especially for 
visualization purposes. 

Autoencoders also act as feature detectors, and they can be used for unsupervised 
pretraining of deep neural networks.

Autoencoder as Generative Models
--------------------------------
Lastly, some autoencoders are generative models: they are capable of randomly generating 
new data that looks very similar to the training data. For example, you could train an 
autoencoder on pictures of faces, and it would then be able to generate new faces. 
However, the generated images are usually fuzzy and not entirely realistic.
In contrast, faces generated by generative adversarial networks (GANs) are now so
convincing that it is hard to believe that the people they represent do not exist.

GANs are now widely used for super resolution (increasing the resolution of
an image), colorization, powerful image editing (e.g., replacing background), 
predicting the next frames in a video, augmenting a dataset (to train other models), 
generating other types of data (such as text, audio, and time series), etc.

Autoencoders vs GANs
---------------------
Autoencoders and GANs are both unsupervised, they both learn dense representations,
they can both be used as generative models, and they have many similar applications.
However, they work very differently:
    
• Autoencoders simply learn to copy their inputs to their outputs. This may sound
like a trivial task, but we will see that constraining the network in various ways
can make it rather difficult. For example, you can limit the size of the latent 
representations, or you can add noise to the inputs and train the network to recover
the original inputs. These constraints prevent the autoencoder from trivially
copying the inputs directly to the outputs, which forces it to learn efficient ways
of representing the data. In short, the codings are byproducts of the autoencoder
learning the identity function under some constraints.

• GANs are composed of two neural networks: a generator that tries to generate
data that looks similar to the training data, and a discriminator that tries to tell
real data from fake data. This architecture is very original in Deep Learning in
that the generator and the discriminator compete against each other during
training: the generator is often compared to a criminal trying to make realistic
counterfeit money, while the discriminator is like the police investigator trying to
tell real money from fake. Adversarial training (training competing neural networks)
is widely considered as one of the most important ideas in recent years. In
2016, Yann LeCun even said that it was “the most interesting idea in the last 10
years in Machine Learning.”

How Auto encoder works
----------------------
Expert chess players are able to memorize the positions of all the pieces in a game by
looking at the board for just five seconds, a task that most people would find impossible.
However, this is only the case when the pieces are placed in realistic positions
(from actual games), not when the pieces were placed randomly. Chess experts don’t
have a much better memory than you and I; they just see chess patterns more easily,
thanks to their experience with the game. Noticing patterns helps them store information
efficiently.
Just like the chess players in this memory experiment, an autoencoder looks at the
inputs, converts them to an efficient latent representation, and then spits out something
that (hopefully) looks very close to the inputs. 

An autoencoder is always composed of two parts: 
    an encoder (or recognition network) that converts the inputs to a latent representation, 
    followed by a decoder (or generative network) that converts the internal representation 
    to the outputs.
    
As you can see, an autoencoder typically has the same architecture as a Multi-Layer
Perceptron (MLP; see Chapter 10), except that the number of neurons in the output
layer must be equal to the number of inputs. In this example, there is just one hidden
layer composed of two neurons (the encoder), and one output layer composed of
three neurons (the decoder). The outputs are often called the reconstructions because
the autoencoder tries to reconstruct the inputs, and the cost function contains a
reconstruction loss that penalizes the model when the reconstructions are different
from the inputs.
Because the internal representation has a lower dimensionality than the input data (it
is 2D instead of 3D), the autoencoder is said to be undercomplete. An undercomplete
autoencoder cannot trivially copy its inputs to the codings, yet it must find a way to
output a copy of its inputs. It is forced to learn the most important features in the
input data (and drop the unimportant ones).    
'''

# Common imports
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
If the autoencoder uses only linear activations and the cost function is the mean
squared error (MSE), then it ends up performing Principal Component Analysis (PCA).

The following code builds a simple linear autoencoder to perform PCA on a 3D dataset,
projecting it to 2D
'''
# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


#Build 3D dataset:
X_train = utils.generate_3d_data(60)
X_train = X_train - X_train.mean(axis=0, keepdims=0)

# Now let's build the Autoencoder...
encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
# a model can be used as a layer in another model
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1.5))

history = autoencoder.fit(X_train, X_train, epochs=20)
codings = encoder.predict(X_train)

# visualize the output of the autoencoder’s hidden layer (i.e., the coding layer, on the right).
fig = plt.figure(figsize=(4,3))
plt.plot(codings[:,0], codings[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.title('2D projection with max variance')
plt.show()

# Comparing with the original 3D dataset, you can see that, the autoencoder
# found the best 2D plane to project the data onto, preserving as much variance
# in the data as it could (just like PCA)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(4,3))
ax = Axes3D(fig)
ax.scatter(X_train[:,0], X_train[:, 1], X_train[:, 2],  c = 'r', marker='o')
plt.xlabel("$X$", fontsize=18)
plt.ylabel("$Y$", fontsize=18, rotation=0)
ax.set_zlabel("$Z$", fontsize=18, rotation=0)
plt.title('original 3D dataset')
plt.show()

#You can think of autoencoders as a form of self-supervised learning
#(i.e., using a supervised learning technique with automatically generated
#labels, in this case simply equal to the inputs).





