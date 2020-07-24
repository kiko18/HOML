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

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line (first filter)
filters[3, :, :, 1] = 1  # horizontal line (second filter)

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.axis("off") # Not shown in the book
plt.show()





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
