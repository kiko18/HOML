# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:40:43 2020

@author: BT
"""

'''
So far we have used only TensorFlow’s high level API, tf.keras, but it already got us
pretty far: we built various neural network architectures, including regression and
classification nets, wide & deep nets and self-normalizing nets, using all sorts of techniques,
such as Batch Normalization, dropout, learning rate schedules, and more. In
fact, 95% of the use cases you will encounter will not require anything else than
tf.keras (and tf.data, see Chapter 13). But now it’s time to dive deeper into TensorFlow
and take a look at its lower-level Python API. This will be useful when you need extra
control, to write custom loss functions, custom metrics, layers, models, initializers,
regularizers, weight constraints and more. You may even need to fully control the
training loop itself, for example to apply special transformations or constraints to the
gradients (beyond just clipping them), or to use multiple optimizers for different
parts of the network.
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Let's start by loading and preparing the California housing dataset. 
# We split it into a training set, a validation set and a test set, and finally we scale it.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)



'''
Custom Loss Functions
---------------------
Suppose you want to train a regression model, but your training set is a bit noisy. Of
course, you start by trying to clean up your dataset by removing or fixing the outliers,
but it turns out to be insufficient, the dataset is still noisy. Which loss function should
you use? The mean squared error might penalize large errors too much, so your
model will end up being imprecise. The mean absolute error would not penalize outliers
as much, but training might take a while to converge and the trained model
might not be very precise. This is probably a good time to use the Huber loss (introduced
in Chapter 10) instead of the good old MSE. The Huber loss is not currently
part of the official Keras API, but it is available in tf.keras (just use an instance of the
keras.losses.Huber class). But let’s pretend it’s not there: implementing it is easy as
pie! Just create a function that takes the labels and predictions as arguments, and use
TensorFlow operations to compute every instance’s loss:
'''
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss) #if small take square_loss 
                                                                #otherwise take linear loss

plt.figure(figsize=(8, 3.5))
z = np.linspace(-4, 4, 200)
plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
plt.plot(z, z**2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
plt.gca().axhline(y=0, color='k')
plt.gca().axvline(x=0, color='k')
plt.axis([-4, 4, 0, 4])
plt.grid(True)
plt.xlabel("$z$")
plt.legend(fontsize=14)
plt.title("Huber loss", fontsize=14)
plt.show()


input_shape = X_train.shape[1:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1),
])


model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_valid_scaled, y_valid))

'''
Saving/Loading Models with Custom Objects
-----------------------------------------
Saving a model containing a custom loss function actually works fine, as Keras just
saves the name of the function. However, whenever you load it, you need to provide a
dictionary that maps the function name to the actual function. More generally, when
you load a model containing custom objects, you need to map the names to the
objects:
'''

model.save("my_model_with_a_custom_loss.h5")

model = tf.keras.models.load_model("my_model_with_a_custom_loss.h5",
                                   custom_objects={"huber_fn": huber_fn})

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


# With the current implementation, any error between -1 and 1 is considered “small”.
# But what if we want a different threshold? One solution is to create a function that
# creates a configured loss function
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")

#Unfortunately, when you save the model, the threshold will not be saved. 
#This means that you will have to specify the threshold value when loading the model 
#(note that the name to use is "huber_fn", which is the name of the function we gave Keras, 
#not the name of the function that created it):
# model = tf.keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
#                                    custom_objects={"huber_fn": create_huber(2.0)})

# You can solve this by creating a subclass of the keras.losses.Loss class, and implement
# its get_config() method

'''
- The constructor accepts **kwargs and passes them to the parent constructor,
    which handles standard hyperparameters: the name of the loss and the reduction
    algorithm to use to aggregate the individual instance losses. By default, it is
    "sum_over_batch_size", which means that the loss will be the sum of the
    instance losses, possibly weighted by the sample weights, if any, and then divide
    the result by the batch size (not by the sum of weights, so this is not the weighted
    mean) Other possible values are "sum" and None.
- The call() method takes the labels and predictions, computes all the instance
    losses, and returns them.
- The get_config() method returns a dictionary mapping each hyperparameter
    name to its value. It first calls the parent class’s get_config() method, then adds
    the new hyperparameters to this dictionary (note that the convenient {**x} syntax
    was added in Python 3.5).
'''
class HuberLoss(tf.keras.losses.Loss):
    
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
  
# create the model again    
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                          input_shape=input_shape),
        tf.keras.layers.Dense(1),
])

# wan then use any instance of this class when you compile the model
model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])


model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))

# When you save a model, Keras calls the loss instance’s get_config() method and
#saves the config as JSON in the HDF5 file. When you load the model, it calls the
#from_config() class method on the HuberLoss class: this method is implemented by
#the base class (Loss) and just creates an instance of the class, passing **config to the
#constructor.
model.save("my_model_with_a_custom_loss_class.h5")    

#model = tf.keras.models.load_model("my_model_with_a_custom_loss_class.h5", # TODO: check PR #25956
#                                   custom_objects={"HuberLoss": HuberLoss})

#That’s it for losses! It was not too hard, was it? 
#Well it’s just as simple for custom activation functions, initializers, regularizers, 
#and constraints. Let’s look at these now.

# The Keras API only specifies how to use subclassing to define layers,
# models, callbacks, and regularizers. If you build other components
# (such as losses, metrics, initializers or constraints) using
# subclassing, they may not be portable to other Keras implementations.

'''
Custom Models
-------------
We already looked at custom model classes in Chapter 10 when we discussed the subclassing
API.10 It is actually quite straightforward, just subclass the keras.models.Model class, 
create layers and variables in the constructor, and implement the call() method to do 
whatever you want the model to do. For example, suppose you want to build the model 
represented in Figure 12-3:
'''
from tensorflow.keras.layers import Dense
# residual block composed of two dense layers and an addition operation
# input -> dense -> dense=output -> (input+output)
# a residualblock adds its inputs to its outputs
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(n_neurons, activation="elu", kernel_initializer="he_normal")  
                       for _ in range(n_layers)]    #a list of dense layers

    def call(self, inputs):
        Z = inputs                 #the input of the residualblock
        for layer in self.hidden:   
            Z = layer(Z)           #model API (add more layer to the input)
        return inputs + Z          #concatenate input and output
    
# Next, we use the subclassing API to define the model itself  
# input -> dense -> residualblock (x3) -> residualblock -> dense
# residualblock: input -> dense -> dense=output -> (input+output)
class ResidualRegressor(tf.keras.models.Model):
    # create the layers in the constructor and use them to perform the computations 
    # you want in the call() method.
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(30, activation="elu", kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)  #Model API
        Z = self.block2(Z)
        return self.out(Z)


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

X_new_scaled = X_test_scaled

model = ResidualRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=5)
score = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_new_scaled)

model.save("my_custom_model.ckpt")

# Load the model and continue training
model = tf.keras.models.load_model("my_custom_model.ckpt")
history = model.fit(X_train_scaled, y_train, epochs=5)    