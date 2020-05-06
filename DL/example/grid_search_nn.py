# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:53:42 2020

@author: BT
"""

# Use scikit-learn to grid search hyperparams
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_mse(y_pred, y_true, verbose=0):
    y_error_0 = np.square(y_pred[:,0] - y_true[:,0])

    mse = np.sum(y_error_0)/(1*y_error_0.shape[0])        
    return mse


housing = fetch_california_housing()

import pandas as pd
df1 = pd.DataFrame(data=housing.data, columns= housing.feature_names)
df1.hist(bins=50, figsize=(20,15))

#scale the features
scaler = StandardScaler()
features = scaler.fit_transform(housing.data)
target = np.expand_dims(housing.target, 1)

df2 = pd.DataFrame(data=features, columns= housing.feature_names)
df2.hist(bins=50, figsize=(20,15))

X_train_full, X_test, y_train_full, y_test = train_test_split(features, target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

del X_train_full, y_train_full, df1, df2



tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]

def build_model(n_neurons=[50,30], learning_rate=3e-3):
    model = Sequential()
    model.add(InputLayer(input_shape=in_dim))
    for n in n_neurons:
        model.add(Dense(n, activation="relu"))
    model.add(Dense(out_dim))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model

keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {
    "n_neurons": [[50,30,15,7]],
    "learning_rate": np.arange(0.0009, 0.0010, 0.0001),
}

rnd_search_cv = GridSearchCV(keras_reg, param_distribs, cv=3, verbose=2)
early_stop = EarlyStopping(monitor='val_loss', patience=40)

rnd_search_cv.fit(X_train, y_train, epochs=2,
                validation_data=(X_valid, y_valid),
                #callbacks=[early_stop]
                )


print('\n \n Best Params: ',rnd_search_cv.best_params_)
print('Best score: ',rnd_search_cv.best_score_)
print('Best estimator: \n \n',rnd_search_cv.best_estimator_)
rnd_search_cv.score(X_test, y_test)

best_model = rnd_search_cv.best_estimator_.model
best_model.summary()
mse_test = best_model.evaluate(X_test, y_test)

y_pred = best_model.predict(X_test)#, batch_size=1)
mse_test_nn = compute_mse(y_pred, y_test)
best_model.save("best_model__.h5")

