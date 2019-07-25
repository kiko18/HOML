# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:26:03 2019

@author: btousside
"""

'''
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

# instanciate the classifiers
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')

# train and evaluate the classifiers. 
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
When this sampling (of training set) is performed with replacement (bootstraping), this method is called
bagging, otherwise it is called pasting.
The following code train an enssemble of 500 Decision trees classifiers, each trained on
100 instances randomly sampled from the training set with replecement (bootstrap = false),
if you set bootstrap to true, it will without replecement (pasting)
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), 
                            n_estimators=500,   # train 500 decision trees
                            max_samples=100,    # ech decision tree is trained on 100 instances
                            bootstrap=True,     # sampling is done with replacement
                            n_jobs=-1,          # number of cpu core to use for training and predictions
                                                # -1 tell sklearn to use all available cores
                            oob_score=True,     # request an automatic out-of-bos evaluation after training
                                                # In fact, since a predictor never sees the oob instances
                                                # during training, it can be evaluated on these instances,
                                                # without the need for a separate validation set or cross-val
                            random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# evaluate the enssemble classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# get the oob evaluation (it should be close to the enssemble classifier)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

#we also have acess to the out of bag (oob) decision function for each instance
#in the case, since the base estimator (tree) can estimate class probabilities
#(i.e it has a predict proba() method)  the decision function return class probabilities
#for each training instance.  For example, the oob evaluator estimates that 
#the first training instance has a 68.25% proba of belonging to the positive class
print('oob decision function for each instance: ', bag_clf.oob_decision_function_)

# see how a single tree will have done
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))


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








