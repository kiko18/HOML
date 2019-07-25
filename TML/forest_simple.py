import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

'''
Random Forest or more generally Enssemble Methods are used near the end of the project,
once we have already built a few good predictors, to combine them into an even better predictor.
Let first build a good decision tree predictor using a gridsearch over some hyperparms.
After that we will then grow a forest combining this predictor to get a more powerfull one
'''

#load moons data and split 
Xm, ym = make_moons(n_samples=1000, noise=0.25, random_state=53)
Xm_train, Xm_test, ym_train, ym_test = train_test_split( Xm, ym, test_size=0.2, random_state=42)

#find the best tree classifier for some hyperparams using a grid search
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)
#fit 882 models =  98 (max_leaf_nodes values) *3 (min_samples_split values)*3(cross val)
grid_search_cv.fit(Xm_train, ym_train)  
# get the best classifier
clf_best = grid_search_cv.best_estimator_
# predict using best classifier
ym_pred = clf_best.predict(Xm_test)
#compute accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(ym_test, ym_pred)

'''
We will train a group of Decision Tree classifiers, each on on a different RANDOM
subset of the training set.
To make predictions, we just obtain the prediction of all individual trees, then
predict the class that gets the most votes. 
Such an enssemble of decision tree is called a RANDOM forest (random because we 
train on random subset o the training set).
Despite its simplicity, random forest is one the most powerful Machine Learning algorithms
available today.
'''


#generate 1000 subsets of the training set, each containing 100 instances selected randomly
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100
mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(Xm_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(Xm_train):
    X_mini_train = Xm_train[mini_train_index]
    y_mini_train = ym_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))


# make a forest consisting of 1k trees, where each tree is a copy of the best classifier find above
from sklearn.base import clone
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

#train one tree of the forest on each (of the 1000 training instance) subset
#and evaluate it on the test set. The overall accuracy is the mean of all accuracies
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)  
    y_pred = tree.predict(Xm_test)
    accuracy_scores.append(accuracy_score(ym_test, ym_pred))

forest_accuracy = np.mean(accuracy_scores)
print('forest_accuracy as mean = ', forest_accuracy)


#Now come the magic: each tree of the forest predict each training instances
# and we keep only the most frequently prediction  
Y_pred_forest = np.empty([n_trees, len(Xm_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred_forest[tree_index] = tree.predict(Xm_test)

# we keep only the most frequently prediction    
from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred_forest, axis=0)

forest_accuracy_with_majority_votes = accuracy_score(ym_test, np.squeeze(y_pred_majority_votes))
print('forest_accuracy with majority votes =', forest_accuracy_with_majority_votes)
   

'''
as we see, the technique with majority vote perform best, raising the accuracy of about 0.5 to 1.5%
'''    
    
    
    
    
    
    
    
    