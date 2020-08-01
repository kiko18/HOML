# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:23:20 2020

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

import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
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


'''
Memory Requirements
-------------------
Another problem with CNNs is that the convolutional layers require a huge amount of RAM. 
This is especially true during training, because the reverse pass of backpropagation
requires all the intermediate values computed during the forward pass.
For example, consider a convolutional layer with 5 × 5 filters, outputting 200 feature
maps of size 150 × 100, with stride 1 and SAME padding. If the input is a 150 × 100
RGB image (three channels), then the number of parameters is (5 × 5 × 3 + 1) × 200
= 15,200 (the +1 corresponds to the bias terms), which is fairly small compared to a
fully connected layer. (A fully connected layer with 150 × 100 neurons, each connected to all 
150 × 100 × 3 inputs, would have 1502 × 1002 × 3 = 675 million parameters!)

However, each of the 200 feature maps contains 150 × 100 neurons,
and each of these neurons needs to compute a weighted sum of its 5 × 5 × 3 = 75 inputs: 
that’s a total of 225 million float multiplications. Not as bad as a fully connected
layer, but still quite computationally intensive. Moreover, if the feature maps
are represented using 32-bit floats, then the convolutional layer’s output will occupy
200 × 150 × 100 × 32 = 96 million bits (12 MB) of RAM. (In the international system of units 
(SI), 1 MB = 1,000 kB = 1,000 × 1,000 bytes = 1,000 × 1,000 × 8 bits.)

And that’s just for one instance! If a training batch contains 100 instances, 
then this layer will use up 1.2 GB of RAM!
During inference (i.e., when making a prediction for a new instance) the RAM occupied
by one layer can be released as soon as the next layer has been computed, so you
only need as much RAM as required by two consecutive layers. But during training
everything computed during the forward pass needs to be preserved for the reverse
pass, so the amount of RAM needed is (at least) the total amount of RAM required by
all layers.

If training crashes because of an out-of-memory error, you can try
reducing the mini-batch size. Alternatively, you can try reducing
dimensionality using a stride, or removing a few layers. Or you can
try using 16-bit floats instead of 32-bit floats. Or you could distribute
the CNN across multiple devices.
'''


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
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)

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
444 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
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




'''
Pretrained Models for Transfer Learning
---------------------------------------
If you want to build an image classifier, but you do not have enough training data,
then it is often a good idea to reuse the lower layers of a pretrained model, 
as we discussed in Chapter 11. For example, let’s train a model to classify 
pictures of flowers, reusing a pretrained Xception model. First, let’s load the dataset 
using TensorFlow Datasets (see Chapter 13).
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from functools import partial

tf_flowers_raw, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)

# get some info about the data
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

print(' \n \n dataset_size:', dataset_size, '\n n_classes', n_classes)
print(class_names)



#Unfortunately, there is only a "train" dataset, no test set or validation set, 
#so we need to split the training set. The TF Datasets project provides an API for this. 
#For example, let’s take the first 10% of the dataset for testing, the next 15% for 
#validation, and the remaining 75% for training.

test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
test_set_raw = tfds.load("tf_flowers", split=test_split, as_supervised=True)
valid_set_raw = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
train_set_raw = tfds.load("tf_flowers", split=train_split, as_supervised=True)

plt.figure(figsize=(12, 10))
index = 0
for image, label in train_set_raw.take(9):  #9 = number of image to plot
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

plt.show()



#Next we must preprocess the images. The CNN expects 224 × 224 images, so we need
#to resize them. We also need to run the image through Xception’s 
#preprocess_input() function. 
#Let’s apply this preprocessing function to all 3 datasets, and let’s also shuffle 
#& repeat the training set, and add batching & prefetching to all datasets.

#If you want to perform some data augmentation, you can just change the preprocessing
#function for the training set, adding some random transformations to the training
#images. For example, use tf.image.random_crop() to randomly crop the images, use
#tf.image.random_flip_left_right() to randomly flip the images horizontally, and
#so on (see the notebook for an example).


def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

batch_size = 32
train_set = train_set_raw.shuffle(1000).repeat()
train_set = train_set.map(partial(preprocess, randomize=True)).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# plot some image of the newly generated dataset
plt.figure(figsize=(12, 12))
for X_batch, y_batch in train_set.take(1):
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


#Next let’s load an Xception model, pretrained on ImageNet. 
#    - We exclude the top of the network (by setting include_top=False): 
#      this excludes the global average pooling layer and the dense output layer. 
#    - We then add our own global average pooling layer, based on the output of 
#      the base model, 
#    - followed by a dense output layer with 1 unit per class, using the softmax activation 
#      function. 
#    - Finally, we create the Keras Model:

base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)   #add global avg pooling
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)    #add output layer
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)  #create model

for index, layer in enumerate(base_model.layers):
    print(index, layer.name)



# As explained in Chapter 11, it’s usually a good idea to freeze the weights 
# of the pretrained layers, at least at the beginning of training:
# (Since our model uses the base model’s layers directly, rather than
#the base_model object itself, setting base_model.trainable=False
# would have no effect.)
for layer in base_model.layers:
    layer.trainable = False

# Finally, we can compile the model and start training:
# This will be very slow, unless you have a GPU. If you do not then you should
#run this on colab, using a GPU runtime (it's free!)
optimizer = tf.keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5)

# we evaluate the model on the test set
print(model.evaluate(test_set))


# After training the model for a few epochs, its validation accuracy should reach about
# 75-80%, and stop making much progress. This means that the top layers are now
# pretty well trained, so we are ready to unfreeze all layers (or you could try unfreezing
# just the top ones), and continue training (don’t forget to compile the model when you
# freeze or unfreeze layers). This time we use a much lower learning rate to avoid damaging
# the pretrained weights:
for layer in base_model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                    nesterov=True, decay=0.001)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=40)

# It will take a while, but this model should reach around 95% accuracy on the test set.
# With that, you can start training amazing image classifiers! But there’s more to computer
# vision than just classification. For example, what if you also want to know where
# the flower is in the picture? Let’s look at this now.

# we evaluate the model on the test set
print(model.evaluate(test_set))
