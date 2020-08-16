# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:58:13 2020

@author: BT
"""


'''
Pooling Layer
-------------
Once you understand how convolutional layers work, the pooling layers are quite
easy to grasp. Their goal is to subsample (i.e., shrink) the input image in order to
reduce the computational load, the memory usage, and the number of parameters
(thereby limiting the risk of overfitting).
Just like in convolutional layers, each neuron in a pooling layer is connected to the
outputs of a limited number of neurons in the previous layer, located within a small
rectangular receptive field. You must define its size, the stride, and the padding type,
just like before. However, a pooling neuron has no weights; all it does is aggregate the
inputs using an aggregation function such as the max or mean. in Max pooling layer, 
only the max input value in each receptive field is propagated to the next layer, 
while the other inputs are dropped. A pooling layer typically works on every input channel 
independently, so the output depth is the same as the input depth.

Other than reducing computations, memory usage and the number of parameters, a
max pooling layer also introduces some level of invariance to small translations.
Moreover, max pooling also offers a small amount of rotational invariance and a
slight scale invariance. Such invariance (even if it is limited) can be useful in cases
where the prediction should not depend on these details, such as in classification
tasks.

But max pooling has some downsides: firstly, it is obviously very destructive: even
with a tiny 2 × 2 kernel and a stride of 2, the output will be two times smaller in both
directions (so its area will be four times smaller), simply dropping 75% of the input
values. And in some applications, invariance is not desirable, for example for semantic
segmentation: this is the task of classifying each pixel in an image depending on the
object that pixel belongs to: obviously, if the input image is translated by 1 pixel to the
right, the output should also be translated by 1 pixel to the right. The goal in this case
is equivariance, not invariance: a small change to the inputs should lead to a corresponding
small change in the output.

Implementing a max pooling layer in TensorFlow is quite easy. The following code
creates a max pooling layer using a 2 × 2 kernel. The strides default to the kernel size,
so this layer will use a stride of 2 (both horizontally and vertically). By default, it uses
VALID padding (i.e., no padding at all):
'''

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

def crop(images):
    return images[150:220, 130:250]

china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower]) 

max_pool = tf.keras.layers.MaxPool2D(pool_size=2) #The strides default to the kernel size,
                                                  #so this layer will use a stride of 2

cropped_images = np.array([crop(image) for image in images], dtype=np.float32)
output = max_pool(cropped_images)

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(cropped_images[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(output[0])  # plot the output for the 1st image
ax2.axis("off")
plt.show()

'''
To create an average pooling layer, just use AvgPool2D instead of MaxPool2D. As you
might expect, it works exactly like a max pooling layer, except it computes the mean
rather than the max. Average pooling layers used to be very popular, but people
mostly use max pooling layers now, as they generally perform better. This may seem
surprising, since computing the mean generally loses less information than computing
the max. But on the other hand, max pooling preserves only the strongest feature,
getting rid of all the meaningless ones, so the next layers get a cleaner signal to work
with. Moreover, max pooling offers stronger translation invariance than average
pooling.

One last type of pooling layer that you will often see in modern architectures is the
global average pooling layer. It works very differently: all it does is compute the mean
of each entire feature map (it’s like an average pooling layer using a pooling kernel
with the same spatial dimensions as the inputs). This means that it just outputs a single
number per feature map and per instance. Although this is of course extremely
destructive (most of the information in the feature map is lost), it can be useful as the
output layer, as we will see later in this chapter.
'''

global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(cropped_images)


# It is actually equivalent to this simple Lamba layer, 
# which computes the mean over the spatial dimensions (height and width)
output_global_avg2 = tf.keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))
output_global_avg2(cropped_images)