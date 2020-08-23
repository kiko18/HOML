# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 20:11:39 2020

@author: BT
"""

'''
In general, you won’t have to implement standard models like GoogLeNet or ResNet
manually, since pretrained networks are readily available with a single line of 
code in the keras.applications
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

# Let load the ResNet-50 model pretrained on ImageNet
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

# Before using the model, we first need to ensure that the images have the right size. 
# A ResNet-50 model expects 224 × 224-pixel images.
# note that the tf.image.resize() will not preserve the aspect ratio. 
# If this is a problem, try cropping the images to the appropriate aspect ratio
# before resizing. Both operations can be done in one shot with tf.image.crop_and_resize()
images_resized = tf.image.resize(images, [224, 224])
#images_resized = tf.image.resize_with_pad(images, 224, 224, antialias=True) #alternative
#images_resized = tf.image.resize_with_crop_or_pad(images, 224, 224)
plot_color_image(images_resized[1])
plt.show()

# The pretrained models assume that the images are preprocessed in a specific way. In
# some cases they may expect the inputs to be scaled from 0 to 1, or –1 to 1, and so on.
# Each model provides a preprocess_input() function that you can use to preprocess
# your images. These functions assume that the pixel values range from 0 to 255, so we
# must multiply them by 255 (since earlier we scaled them to the 0–1 range)
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized * 255)
Y_proba = model.predict(inputs)



# As usual, the output Y_proba is a matrix with one row per image and one column per
# class (in this case, there are 1,000 classes). If you want to display the top K predictions,
# including the class name and the estimated probability of each predicted class,
# use the decode_predictions() function. For each image, it returns an array containing
# the top K predictions, where each prediction is represented as an array containing
# the class identifier, its name, and the corresponding confidence score:
top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    for class_id, name, y_proba in top_K[image_index]:
        print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
    print()















