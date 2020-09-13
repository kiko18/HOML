# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 23:38:02 2020

@author: BT
"""

# Common imports
import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import generate_time_series, generate_data, plot_series, plot_learning_curves
# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

def plot_multiple_forecasts(X, Y, Y_pred, label, loss):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="True values")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)
    plt.title(label + '. Loss= ' + str(loss))

'''
So far we have only predicted the value at the next time step, but we could just as
easily have predicted the value several steps ahead by changing the targets appropriately
(e.g., to predict 10 steps ahead, just change the targets to be the value 10 steps
ahead instead of 1 step ahead). But what if we want to predict the next 10 values?
The first option is to use the model we already trained, make it predict the next value,
then add that value to the inputs (acting as if this predicted value had actually occurred),
and use the model again to predict the following value, and so on.
'''

'''
Forecasting the n next timestep one by one
'''
X_train, y_train, X_valid, y_valid, X_test, y_test = generate_data()
epochs = 20

# model to predict value at the next timestep of a serie
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),  #return_sequences is removed here
    tf.keras.layers.Dense(1)    #output layer is a dense layer
])
# the model is trained on the past data (7k instances/serie)
model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid))


# let's predict the next 10 values one by one:
# model predict value at next timestep (t+1) base on timestep 0-t
# value at t+1 is then added to the inputs
# model predict value at t+2, base on timestep 1-t+1, etc
random_state = 43 # not 42, as it would give the first series in the train set
n_steps = 50
n_step_ahead = 10   #number of step ahead we want to predict
n_new_instances = 100
series = generate_time_series(n_new_instances, n_steps + n_step_ahead, random_state)  #new instance/serie
X_new, y_new = series[:, :n_steps], series[:, n_steps:] #we want to predict the next 10 time step
X = X_new
for step_ahead in range(n_step_ahead): #predict value at t+1 - t+10 one by one
    Xtemp= X[:, step_ahead:]
    y_pred_one = model.predict(Xtemp)[:, np.newaxis, :] #predict value at next timestep based on the 50 last timestep
    X = np.concatenate([X, y_pred_one], axis=1) #add predicted value to input

y_pred_onebyone = X[:, n_steps:] #the predicted value are then the last n_steps vaue of X
loss_oneByOne = np.mean(tf.keras.metrics.mean_squared_error(y_new, y_pred_onebyone))
print('-'*30)
print('shape y_pred_onebyone', y_pred_onebyone.shape)
print('loss one by one =', loss_oneByOne)
print('-'*30)
plot_multiple_forecasts(X_new, y_new, y_pred_onebyone, 'one by one', loss_oneByOne)
plt.show()


'''
As you might expect, the prediction for the next step will usually be more accurate
than the predictions for later time steps, since the errors might accumulate.
You can see this effect in the forcast plot.
A better strategy is to predict the timestep all at once.
In this case, the RNN model is a sequence-to-vector model, since its take a 
sequence/serie and output a vector containing the next n timestep
'''

'''
Train an RNN to Forecasting/predict all n next values at once
'''
#We need to regenerate the training sequences since y_train need to have 10 time steps.
series = generate_time_series(10000, n_steps + n_step_ahead)
X_train, y_train = series[:7000, :n_steps], series[:7000, -n_step_ahead:, 0]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -n_step_ahead:, 0]
X_test, y_test = series[9000:, :n_steps], series[9000:, -n_step_ahead:, 0]

# Now we just need the output layer to have 10 units instead of 1
model = tf.keras.models.Sequential([
tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
tf.keras.layers.SimpleRNN(20),
tf.keras.layers.Dense(n_step_ahead)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid))

# After training this model, you can predict the next 10 values at once very easily
y_pred_allAtOnce = model.predict(X_new)       #[n_series, n_step_ahead]
y_pred_allAtOnce = y_pred_allAtOnce[:, :, np.newaxis]   #[n_series, n_step_ahead, dimensionality]
loss_allAtOnce = np.mean(tf.keras.metrics.mean_squared_error(y_new, y_pred_allAtOnce))
print('-'*10)
print('shape y_pred', y_pred_allAtOnce.shape)
print('loss all at once =', loss_allAtOnce)
print('-'*10)
plot_multiple_forecasts(X_new, y_new, y_pred_allAtOnce, 'all at once', loss_allAtOnce)
plt.show()

'''
For one by one forecasting, the MSE loss is about 0.03. 
For all at once, the loss is about 0.012 which is much better.
As already seen, it is a good idea to compare the performance of your (deep) RNN 
with some baseline metric such as a naive predictions (just forecasting that 
the time series  will remain constant for 10 time steps) or with a simple linear model.
'''

# naive predictions (forecasting that the time series  will remain constant)
y_naive_pred = y_new[:, -1:]    #last value of the time serie
y_naive_pred = np.repeat(y_naive_pred, n_step_ahead, axis=1)    # (last value) is repeated
loss_naive = np.mean(tf.keras.metrics.mean_squared_error(y_new, y_naive_pred))
print('loss navie=', loss_naive)
plot_multiple_forecasts(X_new, y_new, y_naive_pred, 'naive forecast', loss_naive)
plt.show()

# simple linear model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[n_steps, 1]),
    tf.keras.layers.Dense(n_step_ahead)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid))

y_linear_pred = model.predict(X_new)
y_linear_pred = y_linear_pred[:, :, np.newaxis] 
loss_linear = np.mean(tf.keras.metrics.mean_squared_error(y_new, y_linear_pred))
print('loss linear=', loss_linear)
plot_multiple_forecasts(X_new, y_new, y_linear_pred, 'simple linear', loss_linear)
plt.show()

'''
The naive approach is terrible, it gives an MSE of about 0.223. The linear model
gives a loss of about 0.0188. This beat the "one by one" forecasting, which gave a loss of 0.03.
However the "all at once" (with a loss of 0.012) is better than the simple linear model.

But we can still do better: indeed, instead of training the model to forecast 
the next 10 values only at the very last time step, we can train it to forecast 
the next 10 values at each and every time step. 
In other words, we can turn this  sequence-to-vector RNN into a sequence-to-sequence RNN. 
The advantage of this technique is that the loss will contain a term for the output 
of the RNN at each and every time step, not just the output at the last time step. 
This means there will be many more error gradients flowing through the model, 
and they won’t have to flow only through time; they will also flow from the output 
of each time step. This will both stabilize and speed up training.

To be clear, at time step 0 the model will output a vector containing the forecasts for
time steps 1 to 10, then at time step 1 the model will forecast time steps 2 to 11, and
so on. So each target must be a sequence of the same length as the input sequence,
containing a 10-dimensional vector at each step. 

To turn the model into a sequence-to-sequence model, we must set return_sequences=True 
in all recurrent layers (even the last one), and we must apply the output Dense layer at 
every time step. Keras offers a TimeDistributed layer for this purpose:
it wraps any layer (e.g., a Dense layer) and applies it at every time step of its
input sequence. 
It does this efficiently, by reshaping the inputs so that each time step is treated as a 
separate instance (i.e., it reshapes the inputs from [batch_size, time_steps,input dimensions] 
to [batch size × time steps, input dimensions]. 
In this example, the number of input dimensions is 20 because the previous SimpleRNN layer 
has 20 units), then it runs the Dense layer, and finally it reshapes the outputs back to 
sequences (i.e., it reshapes the outputs from [batch size × time steps, output dimensions] to 
[batch size, time steps, output dimensions]; in this example the number of output dimensions 
is 10, since the Dense layer has 10 units). Here is the updated model:
'''

# until now, the target y_true was of shape [batch_size, n_step_ahead],
# that is for each serie we have the next 10 time step "for the whole serie".
# However, now we want the next 10 timestep at each timestep of the serie.
# That is y_true should now be of shape: [batch_size, time_step, n_step_ahead]
# This means we need to regenerate the data
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
y_train = Y[:7000]
y_valid = Y[7000:9000]
y_test = Y[9000:]

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)

# The Dense layer actually supports sequences as inputs (and even higher-dimensional
#inputs): it handles them just like TimeDistributed(Dense(…)), meaning it is applied
#to the last input dimension only (independently across all time steps). Thus, we could
#replace the last layer with just Dense(10). For the sake of clarity, however, we will
#keep using TimeDistributed(Dense(10)) because it makes it clear that the Dense
#layer is applied independently at each time step and that the model will output a
#sequence, not just a single vector.
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),   #to turn the model to a seq-to-seq
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])

# All outputs are needed during training, but only the output at the last time step is
#useful for predictions and for evaluation. So although we will rely on the MSE over all
#the outputs for training, we will use a custom metric
def last_time_step_mse(Y_true, Y_pred):
    return tf.keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

model.compile(loss="mse", 
              optimizer=tf.keras.optimizers.Adam(lr=0.01), 
              metrics=[last_time_step_mse])

history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid))

series = generate_time_series(1, 50 + 10, random_state=43)
X_new, y_new = series[:, :50, :], series[:, 50:, :]
y_pred_seqToSeq = model.predict(X_new)[:, -1][..., np.newaxis]

loss_seq_seq = np.mean(tf.keras.metrics.mean_squared_error(y_new, y_pred_seqToSeq))

print('-'*10)
print('shape y_pred_seqToSeq', y_pred_seqToSeq.shape)
print('loss loss_seq_seq =', loss_seq_seq)
print('-'*10)

plot_multiple_forecasts(X_new, y_new, y_pred_seqToSeq, 'seq-to-seq', loss_seq_seq)
plt.show()

'''
We get a validation MSE of about 0.006, which is 25% better than the previous model.
You can combine this approach with the first one: just predict the next 10 values
using this RNN, then concatenate these values to the input time series and use the
model again to predict the next 10 values, and repeat the process as many times as
needed. With this approach, you can generate arbitrarily long sequences. It may not
be very accurate for long-term predictions, but it may be just fine if your goal is to
generate original music or text, as we will see in Chapter 16.

Simple RNNs can be quite good at forecasting time series or handling other kinds of
sequences, but they do not perform as well on long time series or sequences. 
Let’s discuss why and see what we can do about it.
'''