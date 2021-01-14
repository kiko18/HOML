# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:06:26 2020

@author: BT
"""

'''
Tout comme les SVM, les arbres de decision (decison Trees) sont des algorithmes 
d'apprentissage automatique polyvalents qui peuvent effectuer tout à la fois des 
taches de classification et de regression, et meme des taches a sortie multiples.
Ce sont des algorithms puissant capable de s'adapter a des jeux de données complexes.

les arbres de decision sont aussi les composants fondamentaux des forets aleatoires
(Random Forest) qui figurent parmi les plus puissants des algorithmes d'apprentissage
disponible de nos jours.
'''
import os
import numpy as np

# To make this notebook's output stable across runs
np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#classes names
classes_names = ['setosa', 'versicolor', 'virgina']

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
print(tree_clf)     #try to understand the parameters


from graphviz import Source
from sklearn.tree import export_graphviz

IMAGES_PATH = 'C:/Users/BT/Documents/GitHub/plot'

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
# or open annaconda prompt and run:
# dot -Tpng iris_tree.dot -o iris_tree.png