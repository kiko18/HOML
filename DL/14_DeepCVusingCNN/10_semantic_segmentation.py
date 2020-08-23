# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 00:38:00 2020

@author: BT
"""

'''
In semantic segmentation, each pixel is classified according to the class of the object it
belongs to (e.g., road, car, pedestrian, building, etc.), as shown in Figure 14-26. Note
that different objects of the same class are not distinguished. For example, all the bicycles
on the right side of the segmented image end up as one big lump of pixels. The
main difficulty in this task is that when images go through a regular CNN, they gradually
lose their spatial resolution (due to the layers with strides greater than 1); so, a
regular CNN may end up knowing that there’s a person somewhere in the bottom left
of the image, but it will not be much more precise than that.

Just like for object detection, there are many different approaches to tackle this problem,
some quite complex. However, a fairly simple solution was proposed in the 2015
paper by Jonathan Long et al. we discussed earlier. The authors start by taking a pretrained
CNN and turning it into an FCN. The CNN applies an overall stride of 32 to
the input image (i.e., if you add up all the strides greater than 1), meaning the last
layer outputs feature maps that are 32 times smaller than the input image. This is
clearly too coarse, so they add a single upsampling layer that multiplies the resolution by 32.

There are several solutions available for upsampling (increasing the size of an image),
such as bilinear interpolation, but that only works reasonably well up to ×4 or ×8.
Instead, they use a transposed convolutional layer. It is equivalent to first stretching
the image by inserting empty rows and columns (full of zeros), then performing a
regular convolution (see Figure 14-27). Alternatively, some people prefer to think of
it as a regular convolutional layer that uses fractional strides (e.g., 1/2 in
Figure 14-27). The transposed convolutional layer can be initialized to perform
something close to linear interpolation, but since it is a trainable layer, it will learn to
do better during training. In tf.keras, you can use the Conv2DTranspose layer.

In a transposed convolutional layer, the stride defines how much
the input will be stretched, not the size of the filter steps, so the
larger the stride, the larger the output (unlike for convolutional layers
or pooling layers).
'''

from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def normalize(X):
    return (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower]) 
batch_size, height, width, channels = images.shape

# Let load the ResNet-50 model pretrained on ImageNet
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

images_resized = tf.image.resize(images, [224, 224])

tf.random.set_seed(42)
X = images_resized.numpy()

# Upsampling using a transposed convolutional layer
conv_transpose = tf.keras.layers.Conv2DTranspose(filters=5, kernel_size=3, strides=2, padding="VALID")
output = conv_transpose(X)
print('input shape:  ', X.shape)
print('output shape: ', output.shape)

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(X[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(normalize(output[0, ..., :3]), interpolation="bicubic")  # plot the output for the 1st image
ax2.axis("off")
plt.show()


'''
This solution is OK, but still too imprecise. To do better, the authors added skip connections
from lower layers: for example, they upsampled the output image by a factor
of 2 (instead of 32), and they added the output of a lower layer that had this double
resolution. Then they upsampled the result by a factor of 16, leading to a total upsampling
factor of 32 (see Figure 14-28). This recovered some of the spatial resolution
that was lost in earlier pooling layers. In their best architecture, they used a second
similar skip connection to recover even finer details from an even lower layer. In
short, the output of the original CNN goes through the following extra steps: upscale×2, 
add the output of a lower layer (of the appropriate scale), upscale ×2, add the output
of an even lower layer, and finally upscale ×8. It is even possible to scale up
beyond the size of the original image: this can be used to increase the resolution of an
image, which is a technique called super-resolution.
'''

def upscale_images(images, stride, kernel_size):
    batch_size, height, width, channels = images.shape
    upscaled = np.zeros((batch_size,
                         (height - 1) * stride + 2 * kernel_size - 1,
                         (width - 1) * stride + 2 * kernel_size - 1,
                         channels))
    upscaled[:,
             kernel_size - 1:(height - 1) * stride + kernel_size:stride,
             kernel_size - 1:(width - 1) * stride + kernel_size:stride,
             :] = images
    return upscaled

def normalize(X):
    return (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))


upscaled = upscale_images(X, stride=2, kernel_size=3)
weights, biases = conv_transpose.weights
reversed_filters = np.flip(weights.numpy(), axis=[0, 1])
reversed_filters = np.transpose(reversed_filters, [0, 1, 3, 2])
manual_output = tf.nn.conv2d(upscaled, reversed_filters, strides=1, padding="VALID")

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 2, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(X[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Upscaled", fontsize=14)
ax2.imshow(upscaled[0], interpolation="bicubic")
ax2.axis("off")
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title("Output", fontsize=14)
ax3.imshow(normalize(manual_output[0, ..., :3]), interpolation="bicubic")  # plot the output for the 1st image
ax3.axis("off")
plt.show()

np.allclose(output, manual_output.numpy(), atol=1e-7)

'''
Once again, many GitHub repositories provide TensorFlow implementations of
semantic segmentation (TensorFlow 1 for now), and you will even find pretrained
instance segmentation models in the TensorFlow Models project. Instance segmentation
is similar to semantic segmentation, but instead of merging all objects of the
same class into one big lump, each object is distinguished from the others (e.g., it
identifies each individual bicycle). At present, the instance segmentation models
available in the TensorFlow Models project are based on the Mask R-CNN architecture,
which was proposed in a 2017 paper: it extends the Faster R-CNN model by
additionally producing a pixel mask for each bounding box. So not only do you get a
bounding box around each object, with a set of estimated class probabilities, but you
also get a pixel mask that locates pixels in the bounding box that belong to the object.

As you can see, the field of Deep Computer Vision is vast and moving fast, with all
sorts of architectures popping out every year, all based on convolutional neural networks.
The progress made in just a few years has been astounding, and researchers are now focusing 
on harder and harder problems, such as 
- adversarial learning (which attempts to make the network more resistant to images 
                        designed to fool it), 
- explainability (understanding why the network makes a specific classification), 
- realistic image generation (which we will come back to in Chapter 17), 
- single-shot learning (a system that can recognize an object after it has seen it just once). 

Some even explore completely novel architectures, such as Geoffrey Hinton’s capsule networks
(I presented them in a couple of videos, with the corresponding code in a notebook). 
Now on to the next chapter, where we will look at how to process sequential data such as
time series using recurrent neural networks and convolutional neural networks.
'''


