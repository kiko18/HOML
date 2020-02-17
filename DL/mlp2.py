# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:55:33 2020

@author: BT
"""

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

# First, we need to load a dataset. We will tackle Fashion MNIST, which is a drop-in
# replacement of MNIST (introduced in Chapter 3). It has the exact same format as
# MNIST (70,000 grayscale images of 28×28 pixels each, with 10 classes), but the
# images represent fashion items rather than handwritten digits, so each class is more
# diverse and the problem turns out to be significantly more challenging than MNIST.
# For example, a simple linear model reaches about 92% accuracy on MNIST, but only
# about 83% on Fashion MNIST.


# When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, one
# important difference is that every image is represented as a 28×28 array rather than a
# 1D array of size 784. Moreover, the pixel intensities are represented as integers (from
# 0 to 255) rather than floats (from 0.0 to 255.0).
                                                                                   
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("shape of X_train_full = ", X_train_full.shape)
print("shape of X_train_full = ", X_train_full.dtype)

# create a validation set
# remember that Gradient Descent works better if the feature are scaled
# We scale the feature (pixel intensities) down to the 0-1 range by dividing them by 255.0 (this also
# converts them to floats)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.



plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()



del X_train_full, y_train_full

# With MNIST, when the label is equal to 5, it means that the image represents the
# handwritten digit 5. Easy. However, for Fashion MNIST, we need the list of class
# names to know what we are dealing with
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

'''
Building a Regression MLP Using the Sequential API
--------------------------------------------------
We will see how to build, train, evaluate and use a classification MLP using the
Sequential API
'''
# Now let’s build the neural network! Here is a classification MLP with two hidden layers:
model = keras.models.Sequential()
# Add a flatten layer whose role is simply to convert each input image into a 1D array
# if it receives input data X, it computes X.reshape(-1, 1). This layer does not have any parameters, 
#it is just there to do some simple preprocessing. Since it is the first layer in the model,
# you should specify the input_shape
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# Next, we add 2 dense layers. Each Dense layer manages its own weight matrix, containing all the
# connection weights between the neurons and their inputs. It also manages a vecItor of bias terms 
#(one per neuron). When it receives some input data, it computes g(XW+b)
model.add(keras.layers.Dense(300, activation="relu")) 
model.add(keras.layers.Dense(100, activation="relu"))
# Finally, we add a Dense output layer with 10 neurons (one per class), using the
# softmax activation function (because the classes are exclusive).
model.add(keras.layers.Dense(10, activation="softmax"))

# Instead of adding the layer one by one, we can just pass a list of layer when creating the model.
model2 = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

print(model.summary)
# Note that Dense layers often have a lot of parameters. For example, the first hidden
#layer has 784 × 300 connection weights, plus 300 bias terms, which adds up to
#235,500 parameters! This gives the model quite a lot of flexibility to fit the training
#data, but it also means that the model runs the risk of overfitting, especially when you
#do not have a lot of training data. We will come back to this later. 
print(model.layers)

hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
# Notice that the Dense layer initialized the connection weights randomly (which is
# needed to break symmetry), and the biases were just initializedto zeros, which is fine

# After a model is created, you must call its compile() method to specify the loss function
# and the optimizer to use. Optionally, you can also specify a list of extra metrics to
# compute during training and evaluation:
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"]) #only for classifier 

# we use the "sparse_categorical_crossentropy" loss because we have sparse labels 
# (i.e., for each instance there is just a target class index, from 0 to 9 in this case), 
# and the classes are exclusive. If instead we had one target probability per class for each instance 
# (such as one-hot vectors, e.g. [0.,0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), 
# then we would need to use the "categorical_crossentropy" loss instead. 
# If we were doing binary classification (with one or more binary labels), then we would use the "sigmoid" 
# (i.e., logistic) activation function in the output layer instead of the "softmax" activation
# function, and we would use the "binary_crossentropy" loss.
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

'''
If the training set was very skewed, with some classes being overrepresented and others
underrepresented, it would be useful to set the class_weight argument when
calling the fit() method, giving a larger weight to underrepresented classes, and a
lower weight to overrepresented classes. These weights would be used by Keras when
computing the loss. If you need per-instance weights instead, you can set the sam
ple_weight argument (it supersedes class_weight). This could be useful for example
if some instances were labeled by experts while others were labeled using a
crowdsourcing platform: you might want to give more weight to the former. You can
also provide sample weights (but not class weights) for the validation set by adding
them as a third item in the validation_data tuple.
''' 

'''
The fit() method returns a History object containing the training parameters (his
tory.params), the list of epochs it went through (history.epoch), and most importantly
a dictionary (history.history) containing the loss and extra metrics it
measured at the end of each epoch on the training set and on the validation set (if any)
'''
# plot the learning curve
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

'''
You can see that both the training and validation accuracy steadily increase during
training, while the training and validation loss decrease. Good! Moreover, the validation
curves are quite close to the training curves, which means that there is not too
much overfitting. In this particular case, the model performed better on the validation
set than on the training set at the beginning of training: this sometimes happens
by chance (especially when the validation set is fairly small). However, the training set
performance ends up beating the validation performance, as is generally the case
when you train for long enough. You can tell that the model has not quite converged
yet, as the validation loss is still going down, so you should probably continue training.
It’s as simple as calling the fit() method again, since Keras just continues training
where it left off (you should be able to reach close to 89% validation accuracy).
'''

'''
If you are not satisfied with the performance of your model, you should go back and
tune the model’s hyperparameters, for example the number of layers, the number of
neurons per layer, the types of activation functions we use for each hidden layer, the
number of training epochs, the batch size (it can be set in the fit() method using the
batch_size argument, which defaults to 32). We will get back to hyperparameter
tuning at the end of this chapter.

Once you are satisfied with your model’s validation accuracy, you should evaluate it 
on the test set to estimate the generalization error before you deploy the model to production. 
You can easily do this using the evaluate() method (it also supports several other arguments, 
such as batch_size or sample_weight, please check the documentation for more details):
'''
model.evaluate(X_test, y_test)

'''
As we saw in Chapter 2, it is common to get slightly lower performance on the test set
than on the validation set, because the hyperparameters are tuned on the validation
set, not the test set (however, in this example, we did not do any hyperparameter tuning,
so the lower accuracy is just bad luck). Remember to resist the temptation to
tweak the hyperparameters on the test set, or else your estimate of the generalization
error will be too optimistic.
'''

#we can use the model’s predict() method to make predictions on new instances.
#Since we don’t have actual new instances, we will just use the first instance of the test set
X_new = X_test[:1]
y_proba = model.predict(X_new)
y_proba.round(2)

'''
The model “believes” x_new is probably ankle boots, but it’s not entirely sure, it might be
sneakers instead. If you only care about the class with the highest estimated
probability (even if that probability is quite low) then you can use the pre
dict_classes() method instead.
'''
y_pred = model.predict_classes(X_new)
np.array(class_names)[y_pred]


'''
Building a Regression MLP Using the Sequential API
--------------------------------------------------

Let’s switch to the California housing problem and tackle it using a regression neural
network. For simplicity, we will use Scikit-Learn’s fetch_california_housing()
function to load the data: this dataset is simpler than the one we used in Chapter 2,
since it contains only numerical features (there is no ocean_proximity feature), and
there is no missing value.
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

'''
Building, training, evaluating and using a regression MLP using the Sequential API to
make predictions is quite similar to what we did for classification. The main differences
are the fact that the output layer has a single neuron (since we only want to
predict a single value) and uses no activation function, and the loss function is the
mean squared error. Since the dataset is quite noisy, we just use a single hidden layer
with fewer neurons than before, to avoid overfitting:
'''
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

'''
Building Complex Models Using the Functional API
-----------------------------------------------
'''
