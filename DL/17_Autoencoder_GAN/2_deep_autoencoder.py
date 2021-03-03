# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:25:15 2021

@author: BT
"""

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
Just like other neural networks we have discussed, autoencoders can have multiple
hidden layers. In this case they are called stacked autoencoders (or deep autoencoders).

Adding more layers helps the autoencoder learn more complex codings. That said,
one must be careful not to make the autoencoder too powerful. Imagine an encoder
so powerful that it just learns to map each input to a single arbitrary number (and the
decoder learns the reverse mapping). Obviously such an autoencoder will reconstruct
the training data perfectly, but it will not have learned any useful data representation
in the process (and it is unlikely to generalize well to new instances).

The architecture of a stacked autoencoder is typically symmetrical with regard to the
central hidden layer (the coding layer).
'''

tf.random.set_seed(42)
np.random.seed(42)

# Load the data
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

#Let's build a stacked Autoencoder with 3 hidden layers 
#and 1 output layer (i.e., 2 stacked Autoencoders).
#we split the autoencoder model into two submodels: the encoder and the decoder.

#The encoder takes 28 × 28–pixel grayscale images, flattens them so that each
#image is represented as a vector of size 784, then processes these vectors through
#two Dense layers of diminishing sizes (100 units then 30 units). 
#For each input image, the encoder outputs a vector of size 30.
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])

#The decoder takes codings of size 30 (output by the encoder) and processes them
#through two Dense layers of increasing sizes (100 units then 784 units), and it
#reshapes the final vectors into 28 × 28 arrays so the decoder’s outputs have the
#same shape as the encoder’s inputs.
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

#When compiling the stacked autoencoder, we use the binary cross-entropy loss
#instead of the mean squared error. We are treating the reconstruction task as a
#multilabel binary classification problem: each pixel intensity represents the probability
#that the pixel should be black. Framing it this way (rather than as a regression
#problem) tends to make the model converge faster.
stacked_ae.compile(loss="binary_crossentropy", 
                   optimizer=keras.optimizers.SGD(lr=1.5), 
                   metrics=[rounded_accuracy])

# Finally, we train the model using X_train as both the inputs and the targets 
#(and similarly, we use X_valid as both the validation inputs and targets).
history = stacked_ae.fit(X_train, X_train, epochs=20,
                         validation_data=(X_valid, X_valid))


'''
Visualizing the Reconstructions
-------------------------------
One way to ensure that an autoencoder is properly trained is to compare the inputs
and the outputs: the differences should not be too significant. Let’s plot a few 
images from the validation set, as well as their reconstructions.
'''
        
utils.show_reconstructions(stacked_ae)
#save_fig("reconstruction_plot")        

# The reconstructions are recognizable, but a bit too lossy. 
# We may need to train the model for longer, or make the encoder and decoder deeper, 
# or make the codings larger. 
# But if we make the network too powerful, it will manage to make perfect
# reconstructions without having learned any useful patterns in the data. 
# For now, let’s go with this model.


'''
Dimentionality reduction
-----------------------
Now that we have trained a stacked autoencoder, we can use it to reduce the dataset’s
dimensionality. For visualization, this does not give great results compared to other
dimensionality reduction algorithms (such as those we discussed in Chapter 8), but
one big advantage of autoencoders is that they can handle large datasets, with many
instances and many features. 

So one strategy is to use an autoencoder to reduce the dimensionality down to a 
reasonable level,  then use another dimensionality reduction algorithm for visualization. 

Let’s use this strategy to visualize Fashion MNIST. First, we use the encoder from our 
stacked autoencoder to reduce the dimensionality down to 30, then we use Scikit-Learn’s 
implementation of the t-SNE algorithm to reduce the dimensionality down to 2 for visualization.
'''
from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)

X_valid_2D = TSNE().fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())

# Now we cam plot the dataset
# The t-SNE algorithm identified several clusters which match the classes reasonably
#well (each class is represented with a different color
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.axis("off")
plt.show()




#Let's make this diagram a bit prettier:
# adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
#save_fig("fashion_mnist_visualization_plot")
plt.show()

