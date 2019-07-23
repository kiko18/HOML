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
they don't require feature scaling or centering at all.
'''
# Common imports
import numpy as np

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
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

'''
We can see that a classifier will do better if we increase the depth of the tree.
since it will therefore capture more information to be able to diferenciate between versicolor and virgina.
If we look at the figure 6.2 we see that a flower with lengh 6cm and width 1.5cm is most likely a virgina,
however, with a depth-2 classifier it will be classified as a versicolor
'''
tree_depth = 3
tree_clf = DecisionTreeClassifier(max_depth=tree_depth, random_state=42) #max_depth set how many row we will have
tree_clf.fit(X, y)
y_pred = tree_clf.predict(X)
fscore_2 = f1_score(y, y_pred, average='weighted') 
print('F1 score (depth-3 classifier) = ', fscore_2)