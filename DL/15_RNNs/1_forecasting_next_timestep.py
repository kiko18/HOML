# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:31:07 2020

@author: BT
"""
# Common imports
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

'''
Forecasting a Time Series
-------------------------
Suppose you are studying the number of active users per hour on your website, or the
daily temperature in your city, or your company’s financial health, measured quarterly
using multiple metrics. In all these cases, the data will be a sequence of one or
more values per time step. This is called a time series. In the first two examples there
is a single value per time step, so these are univariate time series, while in the financial
example there are multiple values per time step (e.g., the company’s revenue, debt,
and so on), so it is a multivariate time series. A typical task is to predict future values,
which is called forecasting. Another common task is to fill in the blanks: to predict (or
rather “postdict”) missing values from the past. This is called imputation. For example,
Figure 15-6 shows 3 univariate time series, each of them 50 time steps long, and
the goal here is to forecast the value at the next time step (represented by the X) for
each of them (Time series forecasting).
For simplicity, we will write a function that generate a time series.
'''
from utils import generate_data, plot_series, plot_learning_curves

# generate a time serie and create a training set, a validation set, and a test set. 
# we want to forecast a single value for each series, the targets are column vectors.
# X is of shape [nb_time_series, time_step, dimensionality]
# X is for example the number of active user in a web site at each time_step
X_train, y_train, X_valid, y_valid, X_test, y_test = generate_data()
print('X_train shape:', X_train.shape, '\n', 'y_train shape:', y_train.shape)


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0], y_label=("$x(t)$" if col==0 else None))
plt.show()

'''
Before we start using RNNs, it is often a good idea to have a few baseline metrics, or
else we may end up thinking our model works great when in fact it is doing worse
than basic models. For example, the simplest approach is to predict the last value in
each series. This is called naive forecasting, and it is sometimes surprisingly difficult to
outperform. In this case, it gives us a mean squared error of about 0.020
'''
# This naive approaches consist of predicting that the last value will occur in future (next timestamp)
y_pred = X_valid[:, -1] 
loss_naive = np.mean(tf.keras.losses.mean_squared_error(y_valid, y_pred))


# plot the first serie as well as the true value (blue) and predicted value (red)
instance_id=5
plot_series(X_valid[instance_id, :, 0], y_valid[instance_id, 0], y_pred[instance_id, 0])
print('-------------------')
print(' Naive Approach')
print('Naive loss =', loss_naive)
print('True value =', y_valid[instance_id, 0])
print('Predicted value =', y_pred[instance_id, 0])
print('-------------------')
plt.show()

'''
Another simple approach is to use a fully connected network. Since it expects a flat
list of features for each input, we need to add a Flatten layer. Let’s just use a simple
Linear Regression model so that each prediction will be a linear combination of the
values in the time series. loss is about 0.004. That’s much better than the naive approach
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[50, 1]),
    tf.keras.layers.Dense(1)    #Linear Regression model so that each prediction will be a linear
                                #combination of the values in the time series
])

model.summary() # Note that each neuron has 1 parameter per input and per time step (plus a bias term) 
model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

loss_lr = model.evaluate(X_valid, y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# the prediction (red) should be better than what the naive approaches predicted
y_pred = model.predict(X_valid)
plot_series(X_valid[instance_id, :, 0], y_valid[instance_id, 0], y_pred[instance_id, 0])
print('-------------------')
print(' Linear regression')
print('loss linear regression =', loss_lr)
print('True value =', y_valid[instance_id, 0])
print('Predicted value =', y_pred[instance_id, 0])
print('-------------------')
plt.show()

'''
Implementing a simple RNN              
'''     #                            ___h_____ 
# create the model                  V         |
# single layer and single neuron X --> neuron --> Y = tanh(w_x*X + w_y*h(previous) + b)

# We do not need to specify the length of the input sequences (unlike in the previous model), 
# since a recurrent neural network can process any number of time steps,
# (this is why we set the first input dimension to None).

# By default, the SimpleRNN layer uses the hyperbolic tangent activation function.

# It works exactly as we saw earlier (in figure 15.3):  For each instance,
# the initial state h(init) is set to 0, and it is passed to a single recurrent neuron, 
# along with the value of the first time step, x(0). 
# The neuron computes a weighted sum of these values and applies the hyperbolic tangent activation
# function to the result, and this gives the first output, y0. 
# In a simple RNN, this output is also the new state h0. This new state is passed to the same 
# recurrent neuron along with the next input value, x(1), and the process is repeated until the 
# last time step. Then the layer just outputs the last value, y_49. 
# All of this is performed simultaneously for every time series.
    
input_shape = [None, 1] #[len of input sequence/serie, number of varibale to predict]
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=input_shape)   #only one recurrent layer
    ])
model.summary()
#By default, recurrent layers in Keras only return the final output. 
#To make them return one output per time step, you must set return_sequences=True, as we will see.

# compile and train
optimizer = tf.keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

# evaluate the model
loss_rnn = model.evaluate(X_valid, y_valid)

# plot the learning curve
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# try to predict
y_pred = model.predict(X_valid)
plot_series(X_valid[instance_id, :, 0], y_valid[instance_id, 0], y_pred[instance_id, 0])
print('-------------------')
print(' RNN')
print('loss linear regression =', loss_rnn)
print('True value =', y_valid[instance_id, 0])
print('Predicted value =', y_pred[instance_id, 0])
print('-------------------')
plt.show()

'''
The RNN loss reaches only 0.014, so it is better than the naive approach but it does not beat 
a simple linear model. Note that for each neuron, a linear model has one parameter per input 
and per time step, plus a bias term (in the simple linear model we used, that’s a total of 
51 parameters). 
In contrast, for each recurrent neuron in a simple RNN, there is just: 
    one parameter per input (X) + per hidden state dimension (h) + bias. 
In this simple RNN, that’s a total of just 3 parameters.

Apparently our simple RNN was too simple to get good performance. So let’s try to
add more recurrent layers!
'''

'''
Deep RNNs
---------
It is quite common to stack multiple layers of cells, as shown in Figure 15-7. 
This gives you a deep RNN.
Implementing a deep RNN with tf.keras is quite simple: just stack recurrent layers. 
In this example, we use three SimpleRNN layers (but we could add any other type of
recurrent layer, such as an LSTM layer or a GRU layer, which we will discuss shortly).
We reaches an MSE of 0.003. We finally managed to beat the linear model!

Make sure to set return_sequences=True for all recurrent layers
(except the last one, if you only care about the last output). If you
don’t, they will output a 2D array (containing only the output of
the last time step) instead of a 3D array (containing outputs for all
time steps), and the next recurrent layer will complain that you are
not feeding it sequences in the expected 3D format.

Note that the last layer (tf.keras.layers.SimpleRNN(1)) is not ideal: 
it must have a single unit because we want to forecast a univariate time series, 
and this means we must have a single output value per time step. However, having 
a single unit means that the hidden state is just a single number. That’s really not much, 
and it’s probably not that useful; presumably, the RNN will mostly use the hidden states 
of the other recurrent layers to carry over all the information it needs from time step 
to time step, and it will not use the final layer’s hidden state very much. 
Moreover, since a SimpleRNN layer uses the tanh activation function by default, 
the predicted values must lie within the range –1 to 1. But what if you want to use another 
activation function? For both these reasons, it might be preferable to replace the output layer 
with a Dense layer: it would run slightly faster, the accuracy would be roughly the same, 
and it would allow us to choose any output activation function we want. 
If you make this change, also make sure to remove return_sequences=True from the second 
(now last) recurrent layer.
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)
])

model.summary()
model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))


# evaluate the model
loss_deepRnn = model.evaluate(X_valid, y_valid)

# plot the learning curve
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# try to predict
y_pred = model.predict(X_valid)
plot_series(X_valid[instance_id, :, 0], y_valid[instance_id, 0], y_pred[instance_id, 0])
print('-------------------')
print(' Deep RNN')
print('loss linear regression =', loss_deepRnn)
print('True value =', y_valid[instance_id, 0])
print('Predicted value =', y_pred[instance_id, 0])
print('-------------------')
plt.show()

'''
Note that the last layer (tf.keras.layers.SimpleRNN(1)) is not ideal: 
it must have a single unit because we want to forecast a univariate time series, 
and this means we must have a single output value per time step. However, having 
a single unit means that the hidden state is just a single number. That’s really not much, 
and it’s probably not that useful; presumably, the RNN will mostly use the hidden states 
of the other recurrent layers to carry over all the information it needs from time step 
to time step, and it will not use the final layer’s hidden state very much. 
Moreover, since a SimpleRNN layer uses the tanh activation function by default, 
the predicted values must lie within the range –1 to 1. But what if you want to use another 
activation function? For both these reasons, it might be preferable to replace the output layer 
with a Dense layer: it would run slightly faster, the accuracy would be roughly the same, 
and it would allow us to choose any output activation function we want. 
If you make this change, also make sure to remove return_sequences=True from the second 
(now last) recurrent layer.
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),  #return_sequences is removed here
    tf.keras.layers.Dense(1)    #output layer is a dense layer
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

# evaluate the model
loss_deepRnn2 = model.evaluate(X_valid, y_valid)

# plot the learning curve
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[instance_id, :, 0], y_valid[instance_id, 0], y_pred[instance_id, 0])
print('-------------------')
print(' Deep RNN')
print('loss linear regression =', loss_deepRnn2)
print('True value =', y_valid[instance_id, 0])
print('Predicted value =', y_pred[instance_id, 0])
print('-------------------')
plt.show()

