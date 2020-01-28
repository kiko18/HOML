#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:59:33 2019

@author: bs
"""

'''
In ML Training a model means searching for a combination of model parameters that minimizes 
a cost function (over the trainig set). It is a search in the model's parameters space.
The more parameter a model has, the more dimensions this space has, and the harder the search is.
Searching for a needle in a 300-dimensional haystack is much trickier than in 3 dimensions.

There are 2 differents way of training models in ML.
- using a direct "closed-form" equation that directely computes the model parameters 
  that best fit the model to the tranning set (ie. the model paramater that minimize 
  the cost function over the traning set)
- Using an iterative optimization approach called Gradient Descent (GD) that gradually 
  tweaks the model parameters to minimize the cost function over the traning set. 
  Eventually converging to the same set of parameters as the first method.
  
We will first look at linear regression, a model capable of fitting linear data.
Later on, we will look at Polinomial regression, which is a more complex model that can fit
non-linear datasets. Since this model has more parameters than linear Regression, it is
more prone to overfitting the traning data. So we will learn to detect whether or not
this is the case using learning curves and we will look at several regularization techniques 
that can reduce the risk of overfitting the traning set.
'''

import numpy as np
import matplotlib.pyplot as plt
'''
Let fist start by looking at the Linear Regression model, one of the simplest models there is.
In Chapter 1, we looked at a simple regression model of life satisfaction: 
    life_satisfac‐ tion = theta_0 + theta_1 × GDP_per_capita.
More generally, a linear model makes a prediction by simply computing a weighted sum of the input 
features, plus a constant called the bias term (also called the intercept term)
 y_pred = theta' * X
Training a model means setting its parameters so that the model best fits the training set.
For this purpose, we first need a measure of how well (or poorly) the model fits the training data.
The most common performance measure of a regression model is the Root Mean Square Error (RMSE).
Therefore, to train a Linear Regression model, you need to find the value of theta that minimizes 
the RMSE. In practice, it is simpler to minimize the Mean Square Error (MSE) than the RMSE, and it
leads to the same result (because the value that minimizes a function also minimizes its square 
root). MSE = (1/m)*sum(y_pred - y)^2 
'''

'''
Normal Equation
'''
# To find the value of theta that minimizes the cost function, there is a closed-form solution,
# in other words, a mathematical equation that gives the result directly. 
# This is called the Normal Equation. theta = (x'x)^-1 * (x'y)
# Note that the Normal Equation may not work if the matrix X'X is not invertible (i.e., singular), 
# such as if m < n or if some features are redundant.
# Anther inconvenient of the NE is that it get very slow when the number of features grows large 
# (e.g., 100,000). On the positive side, it is linear with regards to the number of instances 
# in the training set ( O(m)complexity), so it handle large training sets efficiently,
# provided (under the condition that) it fit in memory.
# Another positive point is that once you have trained your Linear Regression model 
# (using the Normal Equa‐ tion or any other algorithm), predictions are very fast: 
# the computational complexity is linear with regards to both the number of instances you want to 
# make predictions on and the number of features. In other words, making predictions on twice as 
# many instances (or twice as many features) will just take roughly twice as much time.

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# make predictions
X_new = np.array([[0], [1], [2]])
X_new_b = np.c_[np.ones((3, 1)), X_new] # add x0 = 1 to each instance 
y_predict = X_new_b.dot(theta_best)

# plot the prediction
plt.plot(X_new, y_predict, "r--")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# Performing linear regression using Scikit-Learn is quite simple
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('lin_reg.intercept_, lin_reg.coef_:', lin_reg.intercept_, lin_reg.coef_)
lin_reg.predict(X_new)

'''
Gradient Descent
'''
# GD is another ways to train a Linear Regression model, better suited for cases where there are 
# a large number of features, or too many training instances to fit in memory.
# Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions 
# to a wide range of problems. The general idea of Gradient Descent is to tweak parameters 
# iteratively in order to minimize a cost function.

# Suppose you are lost in the mountains in a dense fog; you can only feel the slope of the ground 
# below your feet. A good strategy to get to the bottom of the valley quickly is to go downhill 
# in the direction of the steepest slope. This is exactly what Gradient Descent does: it measures
# the local gradient of the error function with regards to the parameter vector theta, and it goes
# in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!

# Concretely, you start by filling theta with random values (this is called random initialization)
# and then you improve it gradually, taking one baby step at a time, each step attempting to 
# decrease the cost function (e.g., the MSE), until the algorithm converges to a minimum.

# An important param in GD is the size of the steps, determined by the learning rate hyperparameter. 
# If the learning rate is too small, then the algorithm will have to go through many iterations
# to converge, which will take a long time.
# On the other hand, if the learning rate is too high, you might jump across the valley and end up
# on the other side, possibly even higher up than you were before. This might make the algorithm 
# diverge, with larger and larger values, failing to find a good solution.
# Finally, not all cost functions look like nice regular bowls. There may be holes, ridges, 
# plateaus, and all sorts of irregular terrains, making convergence to the minimum very difficult.

# Fortunately, the MSE cost function for a Linear Regression model happens to be a convex function
# which means that if you pick any two points on the curve, the line segment joining them never 
# crosses the curve. This implies that there are no local minima, just one global minimum. 
# It is also a continuous function with a slope that never changes abruptly.
# These two facts have a great consequence: Gradient Descent is guaranteed to approach arbitrarily
# close the global minimum (if you wait long enough and if the learning rate is not too high).

# When using Gradient Descent, you should ensure that all features have a similar scale 
# (e.g., using Scikit-Learn’s StandardScaler class), or else it will take much longer to converge.

# To resume, training a model means searching for a combination of model parameters that minimizes
# a cost function (over the training set). It is a search in the model’s parameter space: 
# the more parameters a model has, the more dimensions this space has, and the harder the search 
# is: searching for a needle in a 300-dimensional haystack is much trickier than in 3 dimensions. 
# Fortunately, since the cost function is convex in the case of Linear Regression, the needle is 
# simply at the bottom of the bowl.

# While the Normal Equation can only perform Linear Regression, GD can be used to train many other models.

'''
Batch Gradient Descent
'''
# To implement Gradient Descent, you need to compute the gradient of the cost function
# with regards to each model parameter θ_j. In other words, you need to calculate how much the 
# cost function will change if you change θ_j just a little bit. This is called a partial 
# derivative. It is like asking “what is the slope of the mountain under my feet if I face 
# east?” and then asking the same question facing north (and so on for all other dimensions,

# d MSE / d θ_j = (2/m) * sum(y_pred^i - y^i) * x^i_j

# Instead of computing these partial derivatives individually, you can use the fllowing Equation 
# to compute them all in one go. The gradient vector, noted ∇θMSE(θ), contains all the partial 
# derivatives of the cost function (one for each model parameter)
# Delta MSE = (2/m)*X' (y_pred - y) with y_pred = X*θ

# Notice that this formula involves calculations over the full training set X, at each GD step! 
# This is why the algorithm is called Batch GD: it uses the whole batch of training
# data at every step (actually, Full GD would probably be a better name). 
# As a result it is terribly slow on very large training sets (but we will see much faster GD 
# algorithms shortly). However, Gradient Descent scales well with the number of features; 
# training a Linear Regression model when there are hundreds of thousands of features is much 
# faster using Gradient Descent than using the Normal Equation or SVD decomposition.

# Once you have the gradient vector, which points uphill, just go in the opposite direction to go 
# downhill. This means subtracting ∇θMSE(θ) from θ. This is where the learning rate η comes into 
# play: multiply the gradient vector by η to determine the size of the downhill step.
# θ(next step)   = θ − η * ∇θ MSE(θ)

theta_path_bgd = [] 

eta = 0.1 # learning rate 
n_iterations = 1000 
m=100

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) 
    theta = theta - eta * gradients
    theta_path_bgd.append(theta)

# let try different value of learning rate
def plot_gradient_descent(eta):
    theta = np.random.randn(2,1)  # random initialization
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        if iteration == n_iterations-1:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, "g--")
            
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        
            
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    
np.random.seed(42)

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(eta=0.1)
plt.subplot(133); plot_gradient_descent(eta=0.5)

plt.show()

# On the left, the learning rate is too low: the algorithm will eventually reach the solution, 
# but it will take a long time. In the middle, the learning rate looks pretty good: in just a few
# iterations, it has already converged to the solution. On the right, the learning rate is too 
# high: the algorithm diverges, jumping all over the place and actually getting further and 
# further away from the solution at every step.

# To find a good learning rate, you can use grid search (see Chapter 2). However, you may want to
# limit the number of iterations so that grid search can eliminate models that take too long to 
# converge.

# You may wonder how to set the number of iterations. If it is too low, you will still be far away
# from the optimal solution when the algorithm stops, but if it is too high, you will waste time 
# while the model parameters do not change anymore. A simple solution is to set a very large 
# number of iterations but to interrupt the algorithm when the gradient vector becomes tiny—
# that is, when its norm becomes smaller than a tiny number ε (called the tolerance)—because this 
# happens when Gradient Descent has (almost) reached the minimum.

'''
Stochastic Gradient Descent
'''
# The main problem with Batch GD is the fact that it uses the whole training set to compute 
# the gradients at every step, which makes it very slow when the training set is large.
# At the opposite extreme, Stochastic Gradient Descent just picks a random instance in the 
# training set at every step and computes the gradients based only on that single instance. 
# Obviously this makes the algorithm much faster since it has very little data to manipulate at 
# every iteration. It also makes it possible to train on huge training sets, since only one 
# instance needs to be in memory at each iteration (SGD can be implemented as an out-of-core 
# algorithm)

# On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much less 
# regular than Batch GD: instead of gently decreasing until it reaches the minimum, the cost 
# function will bounce up and down, decreasing only on average. Over time it will end up very 
# close to the minimum, but once it gets there it will continue to bounce around, never settling 
# down. So once the algorithm stops, the final parameter values are good, but not optimal.

# When the cost function is very irregular (with lot of local minimum), this can actually help 
# the algorithm jump out of local minima, so SGD has a better chance of finding the global minimum
# than BGD does.
# Therefore randomness is good to escape from local optima, but bad because it means that the 
# algorithm can never settle at the minimum. 

# One solution to this dilemma is to gradually reduce the learning rate. 
# The steps start out large (which helps make quick progress and escape local minima), 
# then get smaller and smaller, allowing the algorithm to settle at the global minimum. 
# This process is akin to simulated annealing, an algorithm inspired from the process of 
# annealing in metallurgy where molten metal is slowly cooled down. 
# The function that determines the learning rate at each iteration is called the learning schedule
# If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even 
# end up frozen halfway to the minimum. If the learning rate is reduced too slowly, you may jump 
# around the minimum for a long time and end up with a suboptimal solution if you halt training 
# too early.

theta_path_sgd = []

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # not shown in the book
            y_predict = X_new_b.dot(theta)           # not shown
            style = "b-" if i > 0 else "r--"         # not shown
            plt.plot(X_new, y_predict, style)        # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta) 

plt.plot(X, y, "b.")                                 # not shown
plt.xlabel("$x_1$", fontsize=18)                     # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
plt.axis([0, 2, 0, 15])                              # not shown
plt.show()                                           # not shown

# Note that since instances are picked randomly, some instances may be picked several times 
# per epoch while others may not be picked at all. If you want to be sure that the algorithm goes 
# through every instance at each epoch, another approach is to shuffle the training set 
# (making sure to shuffle the input features and the labels jointly), then go through it instance 
# by instance, then shuffle it again, and so on. However, this generally converges more slowly.

# When using Stochastic Gradient Descent, the training instances must be independent and 
# identically distributed (IID), to ensure that the parameters get pulled towards the global 
# optimum, on average. A simple way to ensure this is to shuffle the instances during training 
# (e.g., pick each instance randomly, or shuffle the training set at the beginning of each epoch).
# If you do not do this, for example if the instances are sorted by label, then SGD will start by 
# optimizing for one label, then the next, and so on, and it will not settle close to the global 
# minimum.

# To perform Linear Regression using SGD with Scikit-Learn, you can use the SGDRegressor class, 
# which defaults to optimizing the squared error cost function.


from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) 
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)


'''
Mini Batch GD
'''

# At each step, instead of computing the gradients based on the full training set (as in Batch GD)
# or based on just one instance (as in Stochastic GD), Mini-batch GD computes the gradients 
# on small random sets of instances called mini- batches. The main advantage of Mini-batch GD 
# over Stochastic GD is that you can get a performance boost from hardware optimization of matrix 
# operations, especially when using GPUs.
# The algorithm’s progress in parameter space is less erratic than with SGD, especially with 
# fairly large mini-batches. As a result, Mini-batch GD will end up walking around a bit closer 
# to the minimum than SGD. But, on the other hand, it may be harder for it to escape from local 
# minima (in the case of problems that suffer from local minima, unlike Linear Regression as we 
# saw earlier).
# However, don’t forget that Batch GD takes a lot of time to take each step, and Stochastic GD 
# and Mini-batch GD would also reach the minimum if you used a good learning schedule.


theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000

def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
        
print("theta mgd: ", theta)        




# plot the path for batch GD, SGD and Mini-batch GD

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)


plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()


'''
Polynomial Regression
'''
# What if your data is actually more complex than a simple straight line? 
# Surprisingly, you can actually use a linear model to fit nonlinear data. 
# A simple way to do this is to add powers of each feature as new features, 
# then train a linear model on this extended set of features. 
# This technique is called Polynomial Regression.

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.title("quadratic_data_plot")
plt.show()

# Clearly, a straight line will never fit this data properly. So let’s use Scikit-Learn’s PolynomialFeatures
# class to transform our training data, adding the square (2nd-degree polynomial) of each feature in the 
# training set as new features (in this case there is just one feature)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

# Not bad: the model estimates y = 0.525 x1^2 + 0.95 x1 + 1.97 when in fact the original
# function was y = 0.5 x1^2 + 1.0 x1 + 2.0 + Gaussian noise.

X_new=np.linspace(-3, 3, 100).reshape(100, 1)   # generate some test data point
X_new_poly = poly_features.transform(X_new)     # compute polynomial features for those new data point
y_new = lin_reg.predict(X_new_poly)             # predict the y value for those data point
plt.plot(X, y, "b.")                            # plot the traning data
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")  # plot the predictions
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.title("quadratic_predictions_plot")
plt.show()

# If you perform high-degree Polynomial Regression, you will likely fit the training
# data much better than with plain Linear Regression. Let For example applies
# a 300-degree polynomial model to the preceding training data, and compares the
# result with a pure linear model and a quadratic model (2nd-degree polynomial).
# Notice how the 300-degree polynomial model wiggles around to get as close as possible
# to the training instances.

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.title("high_degree_polynomials_plot")
plt.show()

# Of course, this high-degree Polynomial Regression model is severely overfitting the
# training data, while the linear model is underfitting it. The model that will generalize
# best in this case is the quadratic model. It makes sense since the data was generated
# using a quadratic model, but in general you won’t know what function generated the
# data, so how can you decide how complex your model should be? How can you tell
# that your model is overfitting or underfitting the data?
# In Chapter 2 you used cross-validation to get an estimate of a model’s generalization
# performance. If a model performs well on the training data but generalizes poorly
# according to the cross-validation metrics, then your model is overfitting. 
# If it performs poorly on both, then it is underfitting. This is one way to tell when a model is
# too simple or too complex.
# Another way is to look at the learning curves: these are plots of the model’s performance
# on the training set and the validation set as a function of the training set size
# (or the training iteration). To generate the plots, simply train the model several times
# on different sized subsets of the training set.

