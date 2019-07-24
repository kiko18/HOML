# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:12:18 2019

@author: BT
"""

'''
Decision trees (like svm) are versatile ML algo that can perform both classification and regression tasks.
They are very poweful algo capable of fitting complex datasets.
They are also the fundamental components of Random Forets, which are amoung the most powerful ML algo
available today.
One of the many qualities of Decision Trees is that they require very little data preparation. In particular,
they don't require feature scaling or centering at all. On the other side, it makes very few assomptions 
about the training data as opposed to linear models for example, which obviously assume that the data is linear.
'''
# Common imports
import numpy as np

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# load iris dataset
#One of the many qualities of Decision Trees is that they require very little data preparation. 
#In particular, they don't require feature scaling or centering at all.
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# use a decition tree classifier to fit the iris data
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42) #max_depth set how many row we will have
tree_clf.fit(X, y)

'''
We can visualize the trained Decision Tree by first using the export_graphviz() method to output a graph
definition file .dot, then we can convert this fiile to a variety of format such as pdf or png using the
dot command-line tool from graphviz package:  dot -Tpng iris_tree.dot -o iris_tree.png
we can also visualize directely in graphviz online.
    
'''
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file='C:/Users/BT/Documents/others/tf/iris_tree.dot',
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

'''
Each node has 4 attributes:
    - samples:  how many training instances the node applies to
                For example, 50 training instances have a petal length smaller than 2.4
        
    - value:    how many training instances of each class the node applies to
    
    - gini:     Impurity of the node. 
                a node is pure (gini=0), if all training instances it applies to belong to the same class.
                The gini impurity measure is computed as follow: G(i) = 1 - sum(square(p_i,k))_k =1,...,n
                n is the number of classes, p_i,k = ratio of class k instances amoung the training instances in
                the i^th node.
                For the depth-2 left node for example gini = 1 - (0/54)^2 - (49/54)^2 - (5/54)^2 = 0.168
                Note that 0/54=0% is the proba for setosa, 49/54=90.74% proba for versicolor, etc

Note that sklearn use CART algorithm, which produces only binary trees: nonleaf node always have two children
(i.e., questions only have yes or no answers). However, other algorithms such as ID3 can produce Decision Trees
with nodes that have more than 2 children.
'''


'''
Estimating class proba
----------------------
A decision Tree can also estimate the probability that an instance belongs to a particular class.
To do so, it traverse the tree to find the leaf node for this instance, then it return the ratio/proba
of training instances of class k in this node, finally the instance is attributed to the class with
the highest ratio.
For example, supose we have find a flower whose petals are 5cm long and 1.5cm wide. 
The correpsonding leaf node is the depth-2 left node, so the decision tree should output following proba:
0% setosa (0/54), 90.7% versicolor (49/54), and 9.3% virginica (5/54). Therefore the predicted class
will then be iris-versicolor.
'''
flower = [[5, 1.5]]             #found flower
tree_clf.predict_proba(flower)  #predict proba for each class
tree_clf.predict(flower)        #predict class to which flower belong

# compute how gut the classifier do on the training data
from sklearn.metrics import f1_score
y_pred = tree_clf.predict(X)
fscore = f1_score(y, y_pred, average='weighted') 
print('F1 score (depth-2 classifier) = ', fscore)



from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)
        

plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.title('decision_tree_decision_boundaries_plot')
plt.show()

'''
we can show that classifier will do better if we increase the depth of the tree.
Since it will therefore capture more information to be able to diferenciate between versicolor and virgina.
If we look at the previous figure we see that a flower with lengh 6cm and width 1.5cm is most likely a virgina,
however, with a depth-2 classifier it will be classified as a versicolor (petal width < 1.75)
'''
tree_depth = 3
tree_clf = DecisionTreeClassifier(max_depth=tree_depth, random_state=42) #max_depth set how many row we will have
tree_clf.fit(X, y)
y_pred = tree_clf.predict(X)
fscore_2 = f1_score(y, y_pred, average='weighted') 
print('F1 score (depth-',tree_depth, ' classifier) = ', fscore_2)


'''
hyperparams
-----------
Decision trees belong to the so called nonparametric model, not because it does not have any parameters
(it often has a lot) but because the number of parameters is not determined prior to training, 
so the model structure is free to stick closely to the training data, and most  likely overfiting it.
To avoid overfiting the training data, we need to restrict the Decision tree freedom during training.
As we already know, this is called regularization. Some parameters, which we often restrict are:
    - max_deph: the maximum depth of the tree
    - min_samples_split: the minimun number of samples a node must have before it can be split
    - min_samples_leaf: the minimun number of samples a leaf node must have
    - max_features: the maximun number of features that are evaluated for spliting at each node
    
In the example above we classified he moons dataset with two diferent decision tree classifier,
one with default hyperparams (no restritions), the other with min_samples_leaf = 4.
As we see, the clf with no restrictions overfit the training data and will not gut generalize
'''


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=53)
Xm_train, Xm_test, ym_train, ym_test = train_test_split( Xm, ym, test_size=0.2, random_state=42)

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm_train, ym_train)
deep_tree_clf2.fit(Xm_train, ym_train)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()

# note that he testing set is to small to produce a generalized view of testing
print('train accuracy deep_tree_clf1= ', deep_tree_clf1.score(Xm_train, ym_train))
print('test accuracy deep_tree_clf1= ', deep_tree_clf1.score(Xm_test, ym_test))
print('train accuracy deep_tree_clf2= ', deep_tree_clf2.score(Xm_train, ym_train))
print('test accuracy deep_tree_clf2= ', deep_tree_clf2.score(Xm_test, ym_test))



'''
we can use a grid search to find the best hyperparams
'''
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)
grid_search_cv.fit(Xm_train, ym_train)
# get the best classifier
clf_best = grid_search_cv.best_estimator_
# predict using best classifier
ym_pred = clf_best.predict(Xm_test)
#compute accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(ym_test, ym_pred)


'''
another example with much more data. 
we can play with the hyperparams to increase the test_set accuracy
'''
# get root directory and add it to the system path
#this will give us acess to the data folder in the parent directory
import os
import sys
currDir = os.getcwd()
rootDir = os.path.abspath(os.path.join(currDir, os.pardir))
sys.path.append(rootDir)

#load the planar dataset
from data import data_utils
X_train, X_test, y_train, y_test, classes = data_utils.load_planar_dataset() 

# Visualize the data:
fig = plt.figure()
plt.scatter(X_train[0, :], X_train[1, :], c=np.squeeze(y_train), s=40, cmap=plt.cm.Spectral);
plt.show()

#instanciate 2 differents classifiers
clf1 = DecisionTreeClassifier(random_state=42) 
clf2 = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)  
#fit the model 
clf1.fit(X_train.T, y_train.T) 
clf2.fit(X_train.T, y_train.T) 
# For each classifier, show the decision boundary and compute accuracy
data_utils.plot_decision_boundary(lambda x: clf1.predict(x), X_train, y_train, 'classifier 1')
print('train accuracy = ', clf1.score(X_train.T, y_train.T))
print('test accuracy = ', clf1.score(X_test.T, y_test.T))

data_utils.plot_decision_boundary(lambda x: clf2.predict(x), X_train, y_train, 'clasifier 2')
print('train accuracy = ', clf2.score(X_train.T, y_train.T))
print('test accuracy = ', clf2.score(X_test.T, y_test.T))





'''
regression Trees
----------------
Decision Trees are also capable of performing regression tasks. 
Let's build a regression tree using sklearn and train it on a noisy quadratic dataset 
'''

# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)

export_graphviz(
        tree_reg,
        out_file='C:/Users/BT/Documents/others/tf/quad_tree.dot',
        feature_names=['X'],
        rounded=True,
        filled=True
    )


'''
Just like for classification tasks, Decision Trees are prone to overfitting when dealing with 
regression tasks. Therefore we must restricts the regressor freedom before training
'''
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("No restrictions", fontsize=14)

plt.subplot(122)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)

plt.show()


'''
Instability
--------
Decision Tree are very simple to understand and interpret, easy to use, versatile and powerfull.
However they have a few limitations. In fact, they love orthogonal decision boundaries (all split
are perpendicular to an axis), which makes them sensitive to training set rotation.
For example the above figure show a very simple linear separable dataset.
On the left, a decision tree can split it easily, while on the right after the dataset is rotated 
by 45 degree, the decision boundary look unnecessary convoluted. although both decision Trees 
fit the training set perfectely, it is very likely that the model on the right will not generalize well.
One way to limit this problem is to use PCA, which often result in a better orientation of he training data.
'''

np.random.seed(6)
Xs = np.random.rand(100, 2) - 0.5
ys = (Xs[:, 0] > 0).astype(np.float32) * 2

angle = np.pi / 4
# https://en.wikipedia.org/wiki/Rotation_matrix
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xsr = Xs.dot(rotation_matrix)

# create two clf, one fiting with rotated data
tree_clf_s = DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs, ys)
tree_clf_sr = DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr, ys)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.subplot(122)
plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.show()



'''
More generally, decision tree are very sensitive to small variations in the training data.
For example, if you just remove the widest iris-versicolor from the iris training set 
(the one with petal 4.8 long and 1.8 wide) you will get a completely different decision boundary.
Random forests can limit this instability by averaging predictions over many trees
'''

#reload iris
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# widest Iris-Versicolor flower
X[(X[:, 1]==X[:, 1][y==1].max()) & (y==1)] 

not_widest_versicolor = (X[:, 1]!=1.8) | (y==2)
X_tweaked = X[not_widest_versicolor]
y_tweaked = y[not_widest_versicolor]

tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_tweaked, y_tweaked)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf_tweaked, X_tweaked, y_tweaked, legend=False)
plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)
plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.text(1.0, 0.9, "Depth=0", fontsize=15)
plt.text(1.0, 1.80, "Depth=1", fontsize=13)
plt.show()






