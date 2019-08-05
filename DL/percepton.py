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

per_clf = Perceptron(max_iter=100, tol=-np.infty, random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])

y_pred_train = per_clf.predict(X)

print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y)) * 100))

plot_decision_boundary(lambda x: per_clf.predict(x), X.T, y, 'sklearn Perceptron')


#------------------------------------------------------------------------------------
#my own perceptron class
class MyPerception():
    
    def __init__(self, num_iterations=1000, learning_rate=0.0001, print_cost = False):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.isParamsIntialized=False
          
    #sigmoid is perfect for binary classification since it output values between 0-1    
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z)) 
        return s
    
    def relu(self, z):
        return np.maximum(0, z)

    def derivative(self, f, z, eps=0.000001):
        return (f(z + eps) - f(z - eps))/(2 * eps)

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim,1))
        b = 0
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        
        self.isParamsIntialized = True       
        return w, b
    
    
    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.sigmoid(np.dot(w.T,X) + b)                        # compute activation
        cost = (-1/m)* np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))     # compute cost
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1/m)*np.dot(X,(A-Y).T)
        db = (1/m)*np.sum(A-Y)   
    
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}  
        
        return grads, cost
        
    
    def fit(self, X, Y):               
        costs = []
        
        w, b = self.initialize_with_zeros(X.shape[0])
        
        #retrieve some hyperparams
        num_iterations = self.num_iterations
        learning_rate = self.learning_rate
        print_cost = self.print_cost
        
        if not self.isParamsIntialized:
            return
        
        for i in range(num_iterations):
            # Cost and gradient calculation 
            grads, cost = self.propagate(w, b, X, Y)   
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule 
            w = w - learning_rate*dw
            b = b - learning_rate*db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                  "b": b}
        
        grads = {"dw": dw,
                 "db": db}   
        
        self.params = params
        self.grads = grads
        self.costs = costs
        
    
    def predict(self, X):
        #return
        #Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        
        '''
        the perceptron compute a weighted sum of its imput, 
        then apply a logistic function to that sum,
        if the result exceeds a treshold, it outputs the positive class.
        '''
        w = self.params["w"]
        b = self.params["b"]
        
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid(np.dot(w.T,X) + b) 
        A = np.squeeze(A)
                    
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction = (A>0.5).astype(int)
        
        assert(Y_prediction.shape == (m,))
        
        return Y_prediction
    

#with a learning rate of 0.005 the algo performs worse
mp = MyPerception(num_iterations = 2000, learning_rate = 0.05, print_cost = True)
mp.fit(X.T, y.T)

# Predict test/train set examples
y_prediction_train = mp.predict(X.T)

# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y)) * 100))

plot_decision_boundary(lambda x: mp.predict(x.T), X.T, y, 'MyPerceptron')



'''
A perceptron is a linear classifier.
As we will see here it is not capable of performing well on not linear separable data.
We experience this here on the XOR problem. 
However, it turn out that the problem can be solved by stacking up multiple perceptrion
this will be the purporse of the next script, Multiperceptron.
'''
X = np.array([[0, 0],  [5, 5], [5, 0], [0, 5]])
y = np.array([0, 0, 1, 1])

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral);
plt.xlim([-2, 7])
plt.ylim([-2, 7])
plt.show()

mp.fit(X.T, y.T)
y_pred = mp.predict(X.T)
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y)) * 100))
plot_decision_boundary(lambda x: mp.predict(x.T), X.T, y, 'MyPerceptron')