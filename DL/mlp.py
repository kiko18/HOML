# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:00:23 2019

@author: BT
"""
import numpy as np
import matplotlib.pyplot as plt



class MyMLP():
    
    def __init__(self, n_h = 4, num_iterations = 10000, learning_rate=0.01, print_cost=True):
        self.n_h = n_h   #number of hidden layers
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.areParamsIntialized=False


    def sigmoid(self, x):
        s = 1/(1+np.exp(-x))
        return s
    
    
    def layer_sizes(self, X, Y):
        n_x = X.shape[0] # size of input layer
        n_h = self.n_h
        n_y = Y.shape[0] # size of output layer
        return (n_x, n_h, n_y)
    
    
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """   
        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
        
        W1 = np.random.randn(n_h,n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.expand_dims(np.zeros(1), axis=1)
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}  
        return parameters
    
    
    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = self.sigmoid(Z2)
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}     
        return A2, cache
    
    
    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        
        m = Y.shape[1] # number of example
    
        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
        cost = (-1/m)*np.sum(logprobs) 
        
        cost = float(np.squeeze(cost))     # makes sure cost is the dimension we expect. 
                                           # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))
        
        return cost
    
    
    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]
            
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]
        ### END CODE HERE ###
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2 = A2 - Y
        dW2 = (1/m)*np.dot(dZ2,A1.T)
        db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1/m)*np.dot(dZ1,X.T)
        db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads
    
    
    def update_parameters(self, parameters, grads):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        
        # Update rule for each parameter
        W1 = W1 - self.learning_rate*dW1
        b1 = b1 - self.learning_rate*db1
        W2 = W2 - self.learning_rate*dW2
        b2 = b2 - self.learning_rate*db2
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
    
    
    
    def fit(self, X, Y):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        
        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2"
        parameters = self.initialize_parameters(n_x, self.n_h, n_y)
        
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
             
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(X, parameters)
            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2, Y, parameters)
     
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(parameters, cache, X, Y)
     
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.update_parameters(parameters, grads)
            
            # Print the cost every 1000 iterations
            if self.print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, float(cost)))
        
        self.parameters = parameters
        return parameters
    
    
    
    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        if hasattr(self, 'parameters'):
            # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
            A2, cache = self.forward_propagation(X, self.parameters)
            predictions = A2>0.5
            return predictions
        else:
            print("ERROR: you have not call the fit method to train the class")



# get root directory and add it to the system path
#this will give us acess to the data folder in the parent directory
import os
import sys
currDir = os.getcwd()
rootDir = os.path.abspath(os.path.join(currDir, os.pardir))
sys.path.append(rootDir)

from data import data_utils
X_train, X_test, y_train, y_test, classes = data_utils.load_planar_dataset() 

# Visualize the data:
fig = plt.figure()
plt.scatter(X_train[0, :], X_train[1, :], c=np.squeeze(y_train), s=40, cmap=plt.cm.Spectral);
plt.xlabel('feature X1')
plt.ylabel('feature X2')
plt.title("Dataset")
plt.show()


my_mlp = MyMLP(n_h = 7, num_iterations = 10000, learning_rate=0.15, print_cost=True)
my_mlp.fit(X_train, y_train)

pred_train = my_mlp.predict(X_train)
print ('\n Train Accuracy: %d' % float((np.dot(y_train, pred_train.T) + np.dot(1-y_train,1-pred_train.T))/float(y_train.size)*100) + '% \n')

predictions = my_mlp.predict(X_test)
print ('\n Test Accuracy %d' % float((np.dot(y_test, predictions.T) + np.dot(1-y_test,1-predictions.T))/float(y_test.size)*100) + '% \n')


# Plot the decision boundary
data_utils.plot_decision_boundary(lambda x: my_mlp.predict(x.T), X_train, np.squeeze(y_train), 'Train decision boundary')
#data_utils.plot_decision_boundary(lambda x: my_mlp.predict(x.T), X_test, np.squeeze(y_test), 'Test decision boundary')
