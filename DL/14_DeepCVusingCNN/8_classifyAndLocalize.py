# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:49:45 2020

@author: BT
"""

'''
Localizing an object in a picture can be expressed as a regression task, 
as discussed in Chapter 10: to predict a bounding box around the object, 
a common approach is to predict the horizontal and vertical coordinates of the 
object’s center, as well as its height and width. 
This means we have four numbers to predict. It does not require much change to 
the model used for classification; we just need to add a second dense output 
layer with four units (typically on top of the global average pooling layer), 
and it can be trained using the MSE loss.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

oldTensorflow2 = True

'''load data'''
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
    
    
    
'''load and fine tune Xception model '''
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)

# fine tune Xception model
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)  #classification output
loc_output = tf.keras.layers.Dense(4)(avg)  #localization output (regression)
model = tf.keras.models.Model(inputs=base_model.input,outputs=[class_output, loc_output])

# compile 
'''
The MSE often works fairly well as a cost function to train the model (the regression part),
but it is not a great metric to evaluate how well the model can predict bounding boxes. 
The most common metric for this is the Intersection over Union (IoU): the area of overlap
between the predicted bounding box and the target bounding box, divided by the area of their 
union (see Figure 14-23). In tf.keras, it is implemented by the tf.keras.metrics.MeanIoU class.
'''
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.001)
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.8, 0.2], # depends on what you care most about
              optimizer=optimizer, metrics=["accuracy"])

'''
But now we have a problem: the flowers dataset does not have bounding boxes
around the flowers. So, we need to add them ourselves. This is often one of the hardest
and most costly parts of a Machine Learning project: getting the labels. It’s a good
idea to spend time looking for the right tools. To annotate images with bounding
boxes, you may want to use an open source image labeling tool like VGG Image
Annotator, LabelImg, OpenLabeler, or ImgLab, or perhaps a commercial tool like
LabelBox or Supervisely. You may also want to consider crowdsourcing platforms
such as Amazon Mechanical Turk if you have a very large number of images to annotate.
However, it is quite a lot of work to set up a crowdsourcing platform, prepare the
form to be sent to the workers, supervise them, and ensure that the quality of the
bounding boxes they produce is good, so make sure it is worth the effort. If there are
just a few thousand images to label, and you don’t plan to do this frequently, it may be
preferable to do it yourself. Adriana Kovashka et al. wrote a very practical paper24
about crowdsourcing in computer vision. I recommend you check it out, even if you
do not plan to use crowdsourcing.

Let’s suppose you’ve obtained the bounding boxes for every image in the flowers dataset
(for now we will assume there is a single bounding box per image). You then need
to create a dataset whose items will be batches of preprocessed images along with
their class labels and their bounding boxes. Each item should be a tuple of the form
(images, (class_labels, bounding_boxes)). Then you are ready to train your model!

The bounding boxes should be normalized so that the horizontal
and vertical coordinates, as well as the height and width, all range
from 0 to 1. Also, it is common to predict the square root of the
height and width rather than the height and width directly: this
way, a 10-pixel error for a large bounding box will not be penalized
as much as a 10-pixel error for a small bounding box.
'''
def add_random_bounding_boxes(images, labels):
    fake_bboxes = tf.random.uniform([tf.shape(images)[0], 4])
    return images, (labels, fake_bboxes)

def maximum_precisions(precisions):
    return np.flip(np.maximum.accumulate(np.flip(precisions)))

fake_train_set = train_set.take(5).repeat(2).map(add_random_bounding_boxes)
model.fit(fake_train_set, steps_per_epoch=5, epochs=2)

'''
Classifying and localizing a single object is nice, but what if the images contain multiple
objects (as is often the case in the flowers dataset)?
'''