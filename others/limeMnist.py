# !pip install lime  #if colob

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# https://lime-ml.readthedocs.io/en/latest/

import os,sys
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from skimage.segmentation import mark_boundaries
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

def plotTrainingResult(history):
    plt.plot(history.epoch, history.history['loss'], label="train loss")
    plt.plot(history.epoch, history.history['accuracy'], label="train accuracy")
    plt.plot(history.epoch, history.history['val_loss'], label="validation loss")
    plt.plot(history.epoch, history.history['val_accuracy'], label="validation accuracy")
    plt.title('training + validation loss and accuracy')
    plt.legend()
    plt.show()
    
# Example of CNN to tacckle fashion Mnist
# load and preprocess data
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train=np.repeat(X_train[...,np.newaxis], 3, axis=3)
X_valid=np.repeat(X_valid[...,np.newaxis], 3, axis=3)
X_test=np.repeat(X_test[...,np.newaxis], 3, axis=3)
#X_train = X_train[..., np.newaxis]
#X_valid = X_valid[..., np.newaxis]
#X_test = X_test[..., np.newaxis]

#build CNN
from functools import partial

#partial() defines a thin wrapper around the Conv2D class, called DefaultConv2D: 
#it simply avoids having to repeat the same hyperparameter values over and over again.
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = tf.keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 3]),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

# compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

plotTrainingResult(history)

#------------------------------
# Here is a simpler example of the use of LIME for image classification by using Keras (v2 or greater)

#Explanation
#Now let's get an explanation
explainer = lime_image.LimeImageExplainer()

idx = 128
images = [X_test[idx]]
plt.imshow(images[0][:,:,1])
plt.title(str(y_test[idx]))
plt.show()

# hide_color is the color for a superpixel turned OFF. 
# Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels.
# Here, we set it to 0 (in the representation used by inception model, 0 means gray)
# top_labels: produce explanations for the K labels with highest prediction probabilities.
# If None, produce explanations for all K labels.
explanation = explainer.explain_instance(images[0].astype('double'), model.predict, top_labels=5, hide_color=None, num_samples=1000)

# Now let's see the explanation for the top class
# We can see the top 5 superpixels that are most positive towards the class with the rest of the image hidden
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

#-----------------------
imm = images[0][np.newaxis,...]
print(imm.shape)

yy = model.predict(imm)
print(yy)
print(explanation.top_labels)