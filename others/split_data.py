# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:47:14 2021

@author: BT
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def split_dataset(X, y, labels):
    labels.sort()
    
    y_choose = (y == labels[0]) | (y == labels[1]) | (y == labels[2]) # fish
    y_A = y[~y_choose]
    for ind,lab in enumerate(labels):
        lab -=ind
        y_A =[yy-1 if yy > lab else yy for yy in y_A]
                          
    y_B = y[y_choose]
    for ind,yy in enumerate(labels):
        y_B[y_B==yy] = ind
        
    return ((X[~np.squeeze(y_choose)], np.array(y_A)), (X[np.squeeze(y_choose)], np.array(y_B)))


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
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

task_2_ids = [1,4,8]
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train, task_2_ids)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid, task_2_ids)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test, task_2_ids)

trainA = False
if trainA:
  X_train, y_train = (X_train_A, y_train_A)
  X_valid, y_valid = (X_valid_A, y_valid_A)
  X_test, y_test = (X_test_A, y_test_A)
  n_class = 7
else:
  X_train, y_train = (X_train_B, y_train_B)
  X_valid, y_valid = (X_valid_B, y_valid_B)
  X_test, y_test = (X_test_B, y_test_B)
  n_class = 3

#build CNN
from functools import partial

#partial() defines a thin wrapper around the Conv2D class, called DefaultConv2D: 
#it simply avoids having to repeat the same hyperparameter values over and over again.
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = tf.keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
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
    tf.keras.layers.Dense(units=n_class, activation='softmax'),
])

# compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

score = model.evaluate(X_test, y_test)
print('\n score = ', score)

plotTrainingResult(history)

print(np.unique(y_train_B))