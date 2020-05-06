# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:26:03 2019

@author: Basil
"""

'''
voting classifier
-----------------
Supose you've trained a few classifiers (Logistic regression, SVM, random forest, KNN, ect),
each one achieving about 80% accuracy. A very simple way to create an even better classifier
is to agregate the predictions of each classifier and predict the class that gets the most votes.
this majority-vote classifier is called a hard voting classifier.
'''

import numpy as np
np.random.seed(42)  # to make this notebook's output stable across runs

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# we will train 3 classifiers: Random forest, Logreg, SVM
# then we will train a voting classifier combining these 3 classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# instanciate the 3 diverse classifiers as well as the voting clf
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')

#Let's look at each classifier's accuracy on the test set:
#the result will be that the voting classifier slighly outperforms all individual classifiers
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


'''
Bagging and pasting
-------------------
The example above train different predictor on the same dataset.
This is one way to get a diverse set of predictors.
Another way is to train the same predictor on different random subset of the training set.
When this sampling (of training set) is performed with replacement (bootstraping), 
this method is called bagging, otherwise it is called pasting.
The following code train an enssemble of 500 Decision trees classifiers, each trained on
100 instances randomly sampled from the training set with replecement (bootstrap = True),
if you set bootstrap to False, it will be without replecement (pasting).


'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), 
                            n_estimators=500,   # train 500 decision trees
                            max_samples=100,    # ech decision tree is trained on 100 instances
                            bootstrap=True,     # sampling is done with replacement
                            n_jobs=-1,          # number of cpu core to use for training and predictions
                                                # -1 tell sklearn to use all available cores
                            oob_score=True,     # request an automatic out-of-bag evaluation after training.
                                                # In fact, since a predictor never sees the oob instances
                                                # during training, it can be evaluated on these instances,
                                                # without the need for a separate validation set or cross-val
                            random_state=42)    # 

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# evaluate the enssemble classifier
from sklearn.metrics import accuracy_score
print('Bagging classifier', accuracy_score(y_test, y_pred))


'''
out-of-bag
----------
With bagging, some instances may be sampled several times for any given predictor,
while others may not be sampled at all. By default a BaggingClassifier samples m
training instances with replacement (bootstrap=True), where m is the size of the
training set. This means that only about 63% of the training instances are sampled on
average for each predictor.The remaining 37% of the training instances that are not
sampled are called out-of-bag (oob) instances. Note that they are not the same 37%
for all predictors.
Since a predictor never sees the oob instances during training, it can be evaluated on
these instances, without the need for a separate validation set. You can evaluate the
ensemble itself by averaging out the oob evaluations of each predictor.
In Scikit-Learn, you can set oob_score=True when creating a BaggingClassifier to
request an automatic oob evaluation after training.
'''
# get the oob evaluation (it should be close to the enssemble classifier)
bag_clf.fit(X_train, y_train)
print('oob of Bagging classifier', bag_clf.oob_score_)

# see how a single tree will have done
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print('single tree classifier', accuracy_score(y_test, y_pred_tree))

'''
Bootstrapping introduces a bit more diversity in the subsets that each predictor istrained on, 
so bagging ends up with a slightly higher bias than pasting, but this also means that predictors 
end up being less correlated so the ensemble’s variance is reduced. 
Overall, bagging often results in better models, which explains why it is generally preferred. 
However, if you have spare time and CPU power you can use crossvalidation to evaluate both 
bagging and pasting and select the one that works best.
'''



#we also have acess to the out of bag (oob) decision function for each instance
#in the case, since the base estimator (tree) can estimate class probabilities
#(i.e it has a predict proba() method)  the decision function return class probabilities
#for each training instance.  For example, the oob evaluator estimates that 
#the first training instance has a 68.25% proba of belonging to the positive class
#   print('oob decision function for each instance: ', bag_clf.oob_decision_function_)


#plot both decision boudaries (single tree and enssemble)
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
    
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()

'''
The BaggingClassifier class also supports sampling features as well, training each
predictor on a random subset of the input features.
This is controlled by hyperparams: max_features and bootstrap_features, which work
the same way as max_samples and bootstrap_features.
When sampling both training instance and features, the method is called Random Patches.
Sampling only features (bootstrap_features=True, max_features =...) is called Random subspaces.
'''


'''
Random Forest
-------------
Random Forest is an enssemble of Decision Trees, generally trained via bagging method typically
with max_samples = m (training set size). 
For decision trees classifier, instead of building a BaggingClassifier you can instead use a 
RandomForestClassifier, which is more convenient and optimized for decision trees.
(BaggingClassifier class remain useful if you want a bag of something other than Decision Trees)
With a few exception, a RandomForestClassifier has all the hyperparams of a DecisionTreeClassifier 
(to control how the trees are grown) plus all the hyperparams of a BaggingClassifier 
(to control the enssemble itself) except:
    splitter is absent forced to random,
    presort is absent forced to false, 
    max_samples is absent forced to 1.0
    base_estimator is absent forced to DecisionTreeClassifier with provided hyperparams
    
https://sebastianraschka.com/faq/docs/bagging-boosting-rf.html    
The random forest algorithm is actually a bagging algorithm: also here, we draw random bootstrap 
samples from your training set. However, in addition to the bootstrap samples, we also draw random 
subsets of features for training the individual trees; in bagging, we provide each tree with 
the full set of features. Due to the random feature selection, the trees are more independent of 
each other compared to regular bagging, which often results in better predictive performance 
(due to better variance-bias trade-offs), and I’d say that it’s also faster than bagging, because 
each tree learns only from a subset of features.    
'''


'''
The Random Forest algorithm introduces extra randomness when growing trees;
instead of searching for the very best feature when splitting a node (see Chapter 6), it
searches for the best feature among a random subset of features. This results in a
greater tree diversity, which (once again) trades a higher bias for a lower variance,
generally yielding an overall better model.
'''
from sklearn.ensemble import RandomForestClassifier
#instanciate the classifier
rnd_clf = RandomForestClassifier(n_estimators=500,  #500 trees
                                 max_leaf_nodes=16, #each tree limited to maximun 16 nodes
                                 n_jobs=-1,         #all available cpu
                                 random_state=42)

#train the classifier
rnd_clf.fit(X_train, y_train)
#make prediction using the trained classifier
y_pred_rf = rnd_clf.predict(X_test)
#evaluate the classifier
print('random forest accuracy =', accuracy_score(y_test, y_pred_rf))


'''
The previous random forest classifier is roughly equivalent to the following bagging classifier
Except that random forest algo introduices extra randomness when growing trees, in fact, instead 
of searching  for the very best feature when splitting a node, it search for the best feature
among a random subset of features. This result in a greater tree diversity, which trade higher bias
for a lower variance, generally yielding an overall better model.
'''
#instanciate the classifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
                            n_estimators=500,   #500 trees
                            max_samples=1.0,    
                            bootstrap=True,     #resampling with replecement
                            n_jobs=-1,          #all available cpu   
                            random_state=42)
#train the classifier
bag_clf.fit(X_train, y_train)
#make prediction using the trained classifier
y_pred_bg = bag_clf.predict(X_test)
#evaluate the classifier
print('bagging classifier accuracy =', accuracy_score(y_test, y_pred_bg))

# bagging anf random forest produce almost identical predictions
print('percentage of similar prediction: ', np.sum(y_pred_bg == y_pred_rf) / len(y_pred_bg))


'''
Extra-Trees
-----------
Additionally to random subset of features, extra-trees add random features treshholding.
Rather than searching for the best treshhold (like regular decision tree do), it split 
using a random treshold. Once again, this trade more bias for a lower variance.
Extra trees are much fater to train than regular random forests since finding the best treshold
for each feature at every node is one of the most time-consuming tasks of growing trees.
it is hard to tell in advance whether a RandomForestClassifier will perform better or worse than
an ExtraTreesClassifier. generally, the only way to know is to try both and compare them using cross-val
'''
from sklearn.ensemble import ExtraTreesClassifier
#instanciate the classifier
extra_tree_clf = ExtraTreesClassifier(n_estimators=500,  #500 trees
                                      max_leaf_nodes=16, #each tree limited to maximun 16 nodes
                                      n_jobs=-1,         #all available cpu
                                      random_state=42)

#train the classifier
extra_tree_clf.fit(X_train, y_train)
#make prediction using the trained classifier
y_pred_extra_tree = extra_tree_clf.predict(X_test)
# evaluate the classifier
print('extra tree accuracy =', accuracy_score(y_test, y_pred_extra_tree))


'''
Yet another great quality of random forest is that they make it easy to measure the relative 
importance of each feature. The importance of a feature is measured by looking at how much the 
tree nodes that use that feature reduce impurity on average (across the tree in the forest). 
More precisely it is a weighted average, where each node weight is equal to the number of trainng 
samples that are associated with it.
Sklearn compute this score automatically for each feature after training (accessed via 
feature_importances_), then it scale the results such that the sum of all importances is equal to 1.
'''
# load iris data
from sklearn.datasets import load_iris
iris = load_iris()
# train a random forest classifier on iris data
rdf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rdf_clf.fit(iris["data"], iris["target"])
# print importance of each features
for name, score in zip(iris["feature_names"], rdf_clf.feature_importances_):
    print(name, score)
    
# Another example with digit dataset. What are the most important features/pixels?
# Random forests are very handy to get a quick understanding of what features actually matter,
# in particlar if you need to perform feature selection
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


rdf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rdf_clf.fit(mnist["data"], mnist["target"])

import matplotlib as mpl
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot, interpolation="nearest")
    plt.axis("off")

plot_digit(rdf_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rdf_clf.feature_importances_.min(), rdf_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()


'''
Boosting
--------
Boosting refers to any enssemble method that can combine several weak learners into 
a strong learner. The general idea of most boosting methods is to train predictors
sequentially, each trying to correct its predecessor.
'''

'''
AdaBoost
--------
One way for a new predictor to correct its predecessor is to pay a bit more attention
to the training instances that the predecessor underfitted. This results in new predictors
focusing more and more on the hard cases. This is the technique used in adaBoost.

For example, to build an AdaBoost classifier, a first base classifier (such as decision tree)
is trained and used to make predictions on the training set. The relative weight of 
missclassified training instances is then increased. A second classifier is trained 
using the updated weights and again it makes predictions on the training set, weight are
updated and so on
'''

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),    #decision tree Stump
                                                                     #i.e a decision tree with max_depth=1
                                                                     #i.e a single decision node + 2 leaf node
                              n_estimators=200,
                              algorithm="SAMME.R",                   #only 2 classes
                              learning_rate=0.5)
 
# if your adaBoost is overfitting the training set, you can try reducing the number
# of estimators or more strongly regulirizing the base estimator
ada_clf.fit(X_train, y_train)


'''
Gradient Boosting
-----------------
It is the most popular boosting algorithm. Just like Adaboost, Gradient Boosting works by sequentially
adding predictors to an enssemble, each one correcting its predecessor. However, instead of tweaking 
the instance weights at every iteration like Adaboost does, this method tries to fit the new predictor
to the residual errors made by the previous predictor.
Let see an example of gradient boosting used to solve a regression task. The predictor used is 
Decision Trees. In this case, the algorithm is then called Gradient Tree Boosting 
or Gradient Boosted Regression Trees (GBRT)
'''
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# fit a decision tree regressor to the training set
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# train a second predictor on the residual errors made by the first predictor
y1_pred = tree_reg1.predict(X)
y2 = y - y1_pred   #residual error made by the first predictor
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# train a third regressor on the residual errors made by the second predictor
y3 = y2 - tree_reg2.predict(X)  #residual error made by the second predictor
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

# now we have an enssemble containing three trees. It can make prediction on a new 
# instance simply by adding up the predictions of all the trees
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))



def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.show()


'''
a simple way to train GBRT (Gradient Boosting Regressor Tree) enssemble, is to use 
sklearn GradientBoostingRegressor class. Much like the RandomForestRegressor class,
it has hyperparams to control the growth of Trees (max_depth, min_samples_leaf, etc)
as well as params to control the enssemble training (n_estimators = number of trees).
The following code create the same example as the previous one
'''

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2,       #depth of each trees
                                 n_estimators=3,    #number of trees
                                 learning_rate=1.0) #scale the contribution of each tree
                                                    #if you set it to a low value (like 0.1),
                                                    #you will need more trees in the enssemble
                                                    #to fit the training set but the prediction will
                                                    #usually generalize better. This is a regularization
                                                    #technique called shrinkage.
gbrt.fit(X, y)


# fit a gbrt with a low learning rate and a not enaugh estimators (this will underfit)
gbrt_underfit = GradientBoostingRegressor(max_depth=2, n_estimators=2, learning_rate=0.1, random_state=42)
gbrt_underfit.fit(X, y)

# fit a gbrt with a low learning rate and a lot of estimators (actually too many -> overfitting)
gbrt_overfit = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_overfit.fit(X, y)

plt.figure(figsize=(11,4))

plt.subplot(121)
plot_predictions([gbrt_underfit], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt_underfit.learning_rate, gbrt_underfit.n_estimators), fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_overfit], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_overfit.learning_rate, gbrt_overfit.n_estimators), fontsize=14)
plt.show()


'''
So how to find the optimal number of estimator? 
We can use early stopping to do this. 
A simple way to implement this is to use the staged_predict() method. It return an iterator
over the predictions made by the enssemble at each stage of training (with 1 tree, 2 tree, etc)
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

# Train a GBRT enssemble with 120 trees
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

# Measure the validation error at each stage of training to find the optimal number of trees
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1    #indice of best estimator

#train another GBRT enssemble using the optimal number of trees
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)


# plot 2 figurew, one with the validation error and the other with the best moddel predictions
min_error = np.min(errors)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.show()


'''
It is also possible to implement early stopping by actually stopping training early
(instead of training a large number of trees first and then looking back to find the
optimal number). The following code stops training when the validation error does not
improve for 5 iterations in a row. 
To allow incremental training (i.e keep existing fitted tree when the fit method is called)
we have to set the parameter warm_start to True
'''
gbrt = GradientBoostingRegressor(max_depth=2, 
                                 warm_start=True,   # reuse the solution of the previous call to fit 
                                                    # and add more estimators to the ensemble, otherwise, 
                                                    # just erase the previous solution
                                 subsample=0.25,    # fraction of training instances (selected randomly)
                                                    # to be used for training each tree (here 25%)
                                                    # If smaller than 1.0 this results in 
                                                    # Stochastic Gradient Boosting
                                 random_state=42)

min_val_error = float("inf")
error_going_up = 0

for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)   
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping