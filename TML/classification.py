# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:48:20 2019

@author: BT
"""

# Common imports
import numpy as np

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    plt.show()
    
'''
Dataset loaded by scikit-Learn generally have a similar dictionary structure including:
    - a DESCR key describing the dataset
    - a data key containing an array with one row per instance and one column per feature
    - a target key containing an array with the labels
The MNIST data has 70k images and there are 28*28=784 features
'''
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)

#print(mnist.DESCR)
print(mnist.data)
print(mnist.target)
print(mnist.target.shape)
print(mnist.data.shape)

#build feature and target matrix
X, y = mnist["data"], mnist["target"]

#show one sample
some_digit = X[100]
plot_digit(some_digit)
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)


'''
we should always create a test set and set it aside before inspecting the data closely.
The MNIST dataset is actually already split into a training set (the first 60k images)
and a test set (the last 10k images)
'''
split = 60000
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

'''
We also shuffle the training set, this garantee that all cross-validation folds will be similar
(we don't want one fold to be missing some digits). Moreover, some learning algo are sensitive to 
the order of the training instances and they perform poorly if they get many similar instances in a row.
Shuffling the dataset ensures that this won't happen
'''

shuffle_index = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]



#-----------------------------
'''
Binary classifier
'''

'''
Let simplify the problem for now and only try to identify one digit. for example the number 5.
This 5 detector will be an example of a binary classifier, capable of distinguishing between
just 2 classes, 5 and not-5. As classifier we will first use SGD. It has the advantage of being
capable of handling very large datasets efficiently. This is because it deal with training instances
independently, one at a time (which makes it also well suited for online learning)
'''
y_train_5 = (y_train == 5)  #true for all 5s, false for all other digits
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

'''
Performance Measure/classifier metric:
-accuracy
-confusion matrix
-precision score, recall score
-precision_recall_curve
-F1 score
'''

'''
A good way to evaluate a model is to use cross-validation.
Occasionally you will need more control over the cross-validation process than what Sklearn provides 
off-the-shelf. In this cases, you can implement cross-validation yourself; it is actually fairly 
straightforward. The following code does roughly the same thing as Scikit-Learn's cross_val_score() 
function and prints the result
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# StratifiedKFold perform stratified sampling to produce folds that contain a representative ratio of each class.
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):  
    #at each iteration, create a clone of the classifier, train that clone on the training folds 
    #and makes predictions on the test folds
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)  #count the number of correct predictions
    print(n_correct / len(y_pred))  #outputs ratio of correct predictions


'''
now we will use cross_val_score() to evaluate our SGDClassifier model using k-fold cross-validation 
with 3 folds. Remember that K-fold cross-validation means spliting the training set into K-folds, 
then making predictions and evaluate these predictions on each fold using a model trained 
on the remaining folds. Then average the results from those k experiments
'''
from sklearn.model_selection import cross_val_score
cv_score_1 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print('accuracy score = ', np.mean(cv_score_1))

'''
wow the sgd classifier provide above 95% accuracy, is it not great? Well before we get to excited, 
let's loook at a very dumb classifier that just classifies every single image in the not-5 class.
As we will see, this classifier has over 90% accuracy. this is because only about 10% of the images are 5,
So by always guessing that an image is not 5, you will be right about 90% of the time.
This demonstrate why accuracy is generally not the preferred performance measure for classifiers, especially,
when you're dealing with skewed datasets (i.e., when some classes are much frequent than others)
'''

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool) #always predict false 
    
never_5_clf = Never5Classifier()
cv_score_2 = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

'''
Confusion Matrix
----------------
A much better way to evaluate the performance of a classifier is to look at the confusion matrix.
The general idea is to count the number of times instances of class A are classified as class B
To compute the confusion matrix we first need to have a set of predictions, so they can be compared 
to the targets. We could make prediction on the test set, but remember that we don't want to touch it
until the very end of the project, once we have a classifier that we are ready to launch.
there is one tool we can use: cross_val_predict(). Just like cross_val_score(), it performs k-fold
cross-validation, but instead of returning the evaluation scores, it returns the predictions made
on each test fold. This means we have a clean prediction for each instance in the training set 
(clean meaning that the prediction is made by a model that never saw the data during training)
'''

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

'''
Now we're ready to get the confusion matrix.
its general idea is to count the number of times instances of class A are classified as class B.
For example to know the number of times image 5 were classified as 3 we look in the 5th row and 
3th column of the confusion matrix.
Each row in the confusion matrix represents an actual class, while each column represents a 
predicted class. The confusion matrix is as follow
TN FN
FP TP

'''
from sklearn.metrics import confusion_matrix
confMatrix=confusion_matrix(y_train_5, y_train_pred)

'''
precision and recall 
---------------------
The confusion matrix give you a lot of information but sometimes, we may prefer a more concise metric.
An interessting one to look at is the accuracy of the positive predictions; this is called the precision
of the classifier. precision = TP / (TP + FP). 
Howeever, if there is only one TP (5 correctly classified as 5) and no FP (no non-5 classified as 5) 
then we will get a perfect precision (100%), which make precision alone not very useful. 
So precision is typically used along with another metric named recall also called sensitivity or 
true positive rate(TPR); this is the ratio of positive instances that are correctly detected 
by the classifier. recall = TP / (TP + FN)
'''
from sklearn.metrics import precision_score, recall_score
# when the classifier claims an image is 5, it is correct only 75% of time
pres_scor = precision_score(y_train_5, y_train_pred)
# the classifier only detects 82% of 5s
rec_scor = recall_score(y_train_5, y_train_pred)

'''
F1-score
--------
It is often convenient to combine precision and recall into a single metric called F1-score.
In particular if you want to compare 2 classifiers. The F1-score is the harmonic means of 
precision and recall. Harmonic mean give much more weight to low values, in contrast to 
regular mean which treats all values equally. As a result, the classifier will only get a 
high F1-score if both precision and recall are high. F1-score favors classifiers that have
similar precision and recall.
F1-score = 2 x (precision x recall) / (precision + recall)
'''
from sklearn.metrics import f1_score
f1_scor = f1_score(y_train_5, y_train_pred)
print('F1 score = ', f1_scor)


'''
precision and recall tradeoff
-----------------------------
F1-score favors classifier that have similar precision and recall. 
This is not always what you want. In some context you mostly care about precision 
and in other context recall is more important.
For example, if you train a classifier that detect video that are safe for kids,
you would probably prefer a classifier that rejects many good videos (low recall)
but keeps only save ones (high precision). 
On the other hand suppose your training a classifier to detect shoplifters on surveillance images
it is probably fine if your classifier has only 30% precision (the security get some bad alert),
but a high recall (catching almost all shopliefters).

Unfortunately we can't have it both ways: increasing precision reduces recall, and vice versa,
this is called precision/recall tradeoff.
To set the treshold and do decide to give more weight to péither precision or recall, we use the 
sklearn decision_function() instead of prediction(). its return a score for each instance and 
make prediction based on those scores using any threshhold we want
'''
treshhold_1 = 0
treshhold_2 = 200000
y_scores_ = sgd_clf.decision_function([some_digit])
y_some_digit_pred_1 = y_scores_ > treshhold_1
y_some_digit_pred_2 = y_scores_ > treshhold_2

'''
some_digit is actually a 5, the classifier correctly detect it if threshhold is 0,
but it miss it when treshhold is 200,000. This confirm that increasing the threhhold decrease recall.
So, to set the treshhold we want, we compute the score of all training instance using cross_val_predict() 
but we specify that we want that fct to return decision scores instead of prediction
'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")


'''
With those score we can compute precision and recall for all possible threshhold using precision_recall_curve().
We can then plot precision and recall as a functions of theshold values, to see what threshold you can select.
Another way to select a good precision/recall tradeoff is to plot precision directly against recall.
With this later curve, if we decide for example to aims for 90% precision, we look up the first plot (zooming in a bit)
and find to which threshold it correspond. Then we can make prediction using this treshold.
'''
#
from sklearn.metrics import precision_recall_curve #restricted to binary classification
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1.2])
    plt.title('precision and recall versus decision function')
    plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.title('precision versus recall')
    plt.show()


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plot_precision_vs_recall(precisions, recalls)

# if we set a higher treshold we will get a high precision, but the recall will be low. 
#so everytime someone says let's reach 99% precision, you should ask at what recall?
selected_tresh = 3500 # if we decide to have around 90% prediction with a not to low recall (over 60%)
y_train_pred_with_selected_tresh = y_scores > selected_tresh
precision_score(y_train_5, y_train_pred_with_selected_tresh)
recall_score(y_train_5, y_train_pred_with_selected_tresh)




