# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:02:04 2019

@author: BT
"""

'''
in this script we will train a DNN using tensorflow Estimator high-level API
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

#load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#show one training instance
plt.imshow(X_train[22, :, :])
plt.show()

#Scaling the data
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

#split the data into training and validation
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


'''
The simplest way to train an MLP with TensorFlow is to use the high-level Estimator API 
(formerly tf.contrib.learn).
The DNNClassifier class makes it fairly easy to train a Deep NN with any number of hidden layers,
and a softmax output layer to output the estimated class probabilities. 
'''

#give a directory where the model will be saved
modelDir = 'C:/Users/BT/Documents/others/tf/tf_boards/tf_logs/mlp_estimator'


#create set of real value column from the trainingset
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

# Create a DNN classifier with 2 hidden layers (300 and 100 neurons) as well as 1 softmax output layer with 10 neuron
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], 
                                     feature_columns=feature_cols,
                                     model_dir=modelDir,
                                     n_classes=10, 
                                     #optimizer= 
                                     dropout=None,
                                     activation_fn=tf.nn.relu,
                                     batch_norm=False
                                     )

# train the classifier on 40 iterations using batches of 50 instances
input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": X_train}, y=y_train, num_epochs=5, batch_size=50, shuffle=True)
start = time.time()
dnn_clf.train(input_fn=input_fn)
end = time.time()
print("training time: {:.2f}s".format(end - start))

'''
you will be notified that checkpoint for training and evaluation have been saved to 
C:/Users/BT/AppData/Local/Temp/tmpfgi03ucm/model.ckpt
you can visualise those using tensorboard as seen in linReg.py
'''
#cd c:\programdata\anaconda3\lib\site-packages\tensorboard 
#python main.py --logdir=C:\Users\BT\Documents\others\tf\tf_boards\tf_logs\

# see how good our model perform on test data
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)

print(eval_results)


y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]

'''
Under the hoot, the DNNClassifier class creates all the neuron layers, based on the ReLU activation function 
(we can change this by setting the activation_fn hyperparameter). 
The output layer relies on the softmax function and the cost function is cross entropy 
'''