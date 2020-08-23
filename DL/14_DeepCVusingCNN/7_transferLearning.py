# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:43:19 2020

@author: BT
"""

'''
If you load a pretrained model and use it to make prediction, it can only predict 
classes images the model was trained on. If you want to use it for other classes 
of images you can use transfer learning and therefore still benefit from the 
pretrained models.

Another reason for using transfer learning is if you do not have enough training data.
Then using transfer learning allow you to reuse the lower layer of pretrained model.

Here we will train a model to classify flowers, reusing a pretrained Xception model.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

oldTensorflow2 = True
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)


dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5

# split the data: first 10% for testing, the next 15% for validation, 
# and the remaining 75% for training.
if oldTensorflow2:  
    # the way data are split have change in new tensorflow version
    test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
    test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
    valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
    train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)
else:
    test_set, valid_set, train_set = tfds.load("tf_flowers",
                                               split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                               as_supervised=True)


plt.figure(figsize=(12, 10))
index = 0
for image, label in train_set.take(9):
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

plt.show()

'''
Next we must preprocess the images. The Xception CNN expects 224 × 224 images, 
so we need to resize them. We also need to run the images through Xception’s 
preprocess_input() function.
'''

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

# slightly fancier preprocessing
def preprocess__(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# simple preprocessing
#If you want to perform some data augmentation, change the preprocessing function
#for the training set, adding some random transformations to the training images. For
#example, use tf.image.random_crop() to randomly crop the images, use
#tf.image.random_flip_left_right() to randomly flip the images horizontally, and so on
# as done in the fancier preprocessing function above
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# Let’s apply this preprocessing function to all three datasets, 
# shuffle the training set and add batching and prefetching to all the datasets
batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# show cropped images
plt.figure(figsize=(12, 12))
for X_batch, y_batch in train_set.take(1):
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()

'''
That is for data preparation.
The keras.preprocessing.image.ImageDataGenerator class
makes it easy to load images from disk and augment them in various
ways: you can shift each image, rotate it, rescale it, flip it horizontally
or vertically, shear it, or apply any transformation function
you want to it. This is very convenient for simple projects. However,
building a tf.data pipeline has many advantages: it can read
the images efficiently (e.g., in parallel) from any source, not just the
local disk; you can manipulate the Dataset as you wish; and if you
write a preprocessing function based on tf.image operations, this
function can be used both in the tf.data pipeline and in the model
you will deploy to production.
'''

# Next let’s load an Xception model, pretrained on ImageNet. 
# We exclude the top of the network by setting include_top=False. 
# This excludes the global average pooling layer and the dense output layer. 
# We then add our own global average pooling layer, based on the output of the 
# base model, followed by a dense output layer with one unit per class, 
# using the softmax activation function. Finally, we create the Keras Model
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

for index, layer in enumerate(base_model.layers):
    print(index, layer.name)

# As explained in Chapter 11, it’s usually a good idea to freeze the weights 
# of the pretrained layers, at least at the beginning of training.
# This give the new added layer time to learn some reasonable weight
for layer in base_model.layers:
    layer.trainable = False

# Finally, we can compile the model and start training. we first train for only 
# few epoch to give the new added layer time to learn reasonable weight
# Note that this will be very slow, unless you have a GPU. 
# If you do not, you can use Colab, using a GPU runtime (it’s free!).
optimizer = tf.keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5)   #we train only for 5 epochs

# After training the model for a few epochs, its validation accuracy should reach 
# about 75–80% and stop making much progress. This means that the top layers are now
# pretty well trained, so we are ready to unfreeze all the layers (or you could try
# unfreezing just the top ones) and continue training (don’t forget to compile the
# model when you freeze or unfreeze layers). This time we use a much lower learning
# rate to avoid damaging the pretrained weights.
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
                    epochs=40)  ##we now train longer


'''
It will take a while, but this model should reach around 95% accuracy on the test set.
With that, you can start training amazing image classifiers! But there’s more to computer
vision than just classification. For example, what if you also want to know where
the flower is in the picture? Let’s look at this in the next part.
'''












