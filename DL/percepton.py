# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:19:05 2019

@author: BT
"""

# get root directory and add it to the system path
import os
import sys
currDir = os.getcwd()
rootDir = os.path.abspath(os.path.join(currDir, os.pardir))
sys.path.append(rootDir)

from data.data_utils import plot_decision_boundary
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

'''
The perceptron is one of the simplest ANN architectures. 
invented in 1957 by Frank Rosenblatt.
It is based on a slighly different artificial neuron called a 
linear threshold unit (LTU)  h = g(WX) where g is a step function
The most commonly step function used in perceptron is the sign function
 g(z) = 0 if z<0 , 1 if z>=0
 
The LTU works fine on simple linear binary classification.
It compute a linear combination of the input and if the result exceeds a trehshold, 
it output the positive class.

Note that contrary to logistic regression, perceptron doesn't output a class probability,
rather they just makes prediction based on a hard treshhold. That is one of the good 
reason to prefer logReg over Perceptrons.

As they are linear, perceptron fail at solving some trivial problems like XOR classification problem.
Of course this is true for other linear classification model like logReg, but researchers had expected
much more from perceptron. However it turn out that most of the limitations of the perceptron
can be eliminated by stacking multiple perceptrons.
'''

iris = load_iris()
X = iris.data[:, (2,3)] #petal length, petal width
y = (iris.target == 0).astype(np.int)  #iris setosa

# Visualize the data:
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral);
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title("Dataset")
plt.show()

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])


plot_decision_boundary(lambda x: per_clf.predict(x), np.transpose(X), y, '')