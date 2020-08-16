# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:48:59 2020

@author: BT
"""

'''
In the last few years, thanks to the increase in computational power, 
the amount of available training data, and the tricks presented in Chapter 11 
for training deep nets, CNNs have managed to achieve superhuman performance on 
some complex visual tasks. They power image search services, self-driving cars, 
automatic video classification systems, and more.

Why not simply use a regular deep neural network with fully connected
layers for image recognition tasks? Unfortunately, although
this works fine for small images (e.g., MNIST), it breaks down for
larger images because of the huge number of parameters it
requires. For example, a 100 × 100 image has 10,000 pixels, and if
the first layer has just 1,000 neurons (which already severely
restricts the amount of information transmitted to the next layer),
this means a total of 10 million connections. And that’s just the first
layer. CNNs solve this problem using partially connected layers and
weight sharing.

In a CNN, each neuron in the second convolutional layer is connected
only to neurons located within a small rectangle in the first layer. This architecture
allows the network to concentrate on small low-level features in the first hidden layer,
then assemble them into larger higher-level features in the next hidden layer, and so
on. This hierarchical structure is common in real-world images, which is one of the
reasons why CNNs work so well for image recognition.

In order for a layer to have the same height and width as the previous layer, 
it is common to add zeros around the inputs, as shown in the diagram. 
This is called zero padding. It is also possible to connect a large input layer 
to a much smaller layer by spacing out the receptive fields. 
The shift from one receptive field to the next is called the stride.

A layer full of neurons using the same filter outputs a feature map, which highlights 
the areas in an image that activate the filter the most. Of course you do not have to 
define the filters manually: instead, during training the convolutional layer will 
automatically learn the most useful filters for its task, and the layers above will 
learn to combine them into more complex patterns.
'''


from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


    
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")

def crop(images):
    return images[150:220, 130:250]
    
# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower]) 
batch_size, height, width, channels = images.shape

# Create 2 fi
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line (first filter)
filters[3, :, :, 1] = 1  # horizontal line (second filter)

'''
Convolution layer:
-----------------
- images is the input mini-batch. In tf a mini-batch is represented as a 4D tensor of 
  shape [mini-batch size, height, width, channels].
- filters is the set of filters to apply. The weights of a convolutional layer 
  (also the filters) are represented as a 4D tensor of shape [fh, fw, fn′, fn]
- strides is equal to 1, but it could also be a 1D array with 4 elements, where the
  two central elements are the vertical and horizontal strides (sh and sw). The first
  and last elements must currently be equal to 1. They may one day be used to
  specify a batch stride (to skip some instances) and a channel stride (to skip some
  of the previous layer’s feature maps or channels).
- padding must be either "VALID" or "SAME": 
    * If set to "VALID", the convolutional layer does not use zero padding, 
      and may ignore some rows and columns at the bottom and right of the input image, 
      depending on the stride
    * If set to "SAME", the convolutional layer uses zero padding if necessary. 
      In this case, the number of output neurons is equal to the number of input neurons
      divided by the stride, rounded up. Then zeros are added as evenly as possible 
      around the inputs.
'''
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.axis("off") # Not shown in the book
plt.show()


for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
        plot_image(outputs[image_index, :, :, feature_map_index])

plt.show()



plot_image(crop(images[0, :, :, 0]))
plt.show()

for feature_map_index, filename in enumerate(["china_vertical", "china_horizontal"]):
    plot_image(crop(outputs[0, :, :, feature_map_index]))
    plt.show()

plot_image(filters[:, :, 0, 0])
plt.show()
plot_image(filters[:, :, 0, 1])
plt.show()