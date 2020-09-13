# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:30:24 2020

@author: BT
"""

# Common imports
import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# This function creates as many time series as requested (via the batch_size argument),
# each of length n_steps, and there is just one value per time step in each series
# (i.e., all series are univariate). 
# The function returns a NumPy array of shape [batch_size, time steps, 1], where each 
# series is the sum of two sine waves of fixed amplitudes but random frequencies and phases, 
# plus a bit of noise.
# When dealing with time series (and other types of sequences such as sentences), 
# the input features are generally represented as 3D arrays of shape 
# [batch size, time steps, dimensionality], where dimensionality is 1 for univariate time series 

def generate_time_series(batch_size, n_steps, random_state=42):
    np.random.seed(random_state)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    n_steps = len(series)
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)   #last timestemp true value in blue
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")     #last timestemp predicted value in red
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    plt.legend()

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)    


# generate a time serie and create a training set, a validation set, and a test set 
# we want to forecast a single value for each series, the targets are column vectors   
# X is of shape [nb_time_series, time_step, dimensionality]
def generate_data(nb_time_series = 10000, n_steps=50, dimensionality = 1):
    series = generate_time_series(nb_time_series, n_steps + dimensionality) #10k time serie with 50 timesep
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test