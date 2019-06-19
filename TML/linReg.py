# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:09:15 2019

@author: basil
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

'''
This fct create a datasets/housing directory in your workspace, 
download the housing.tgz file and extracts the housing.csv from it
'''
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    

'''
Load the califormia housing data from a repository
'''
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#feach the data from the website and load them using pandas
#fetch_housing_data()
housing = load_housing_data()

#understand your data
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


#randomly divide data into training and test set
def split_train_test(data, test_ratio):
    np.random.seed((42))
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


'''
Supose we have chatted with experts who told us that the median income is a very important attribute 
to predict median housing price. We may want to ensure that the test set is representative of the various
categories of incomes in the whole datasets. Since the median income is a continous numerical attribute, 
we first need to create an income category attribute.

It is important to have sufficient number of instance in your dataset for each stratum.
'''
#create an incoming category attribute
# Divide by 1.5 to limit the number of income categories
#round up using ceil to have discrete categorie
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
print(housing["income_cat"].value_counts())


#Do stratified sampling 
#for a USA data set on gender whit 48% men and 52% femele, 
#the goal of stratified sampling will be to divise the dataset such that the proportion 
#of man an femele are maintaint in both training and testset
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
'''
To see if this worked as expected we can look at the income category proportions in the 
full housing dataset. This should almost identical to the income category proportions 
in the training and test set
'''
print("-----------------------------_")
print(housing["income_cat"].value_counts() / len(housing))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print("-----------------------------_")

'''
If we check these proportion for the test set generated via purely random sampling,
we will see that it is quite skewed
'''
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(housing["income_cat"].value_counts() / len(housing))
print(train_set["income_cat"].value_counts() / len(train_set))
print(test_set["income_cat"].value_counts() / len(test_set))

#stratified sampling is a good way to split our data such that important attribut are equaly distributed
#in the training and test set
#we remove the income_cat so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    

'''
Now we will explore the data a little more in depth.
We will only work with the traning set.
In case the training set is very large, one often create an exploratory set to make 
manipulation easy and fast
'''
#we make a copy of the trainingset so we can explore it without harming the original
housing = strat_train_set.copy()

#in this pot it is hard to see any pattern
housing.plot(kind="scatter", x="longitude", y="latitude")
#here we can see where is a high density area
#namely the bay area and around Los Angeles and San Diego
#as well as a fairly high density in the central valley
#in paticular around sacramento and fresno
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#we now look at the housing prices, represented by the color
#we use a predefined color map (option cmap) called jet, which range from 
#blue(low value) to red(heigh price)
#we can see that the prices are related to the location (close to the ocean)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()

'''
Now we will investigates the correlation  between attributes.
We can compute the standart correlation coefficient (also called pearson's r)
between every pair of attribues using the corr() methods)
Note that the correlation coef only measure linear correlation (if x goes up, y goes up/down)
'''
corr_matrix = housing.corr()

#we can for example look how much each attribute correlates with the median house value
#the correlation coef range from -1 to 1.  1 indicate a strong positive corelation.
#for example, the median house tend to go up when the median income goes up
#-1 indicates a strong negative coefficient. For example between the latitude 
#and the median house value (i.e prices have a slight tendency to goes down when you go north)
#Finally, coef close to zero means that there is no linear corelation
corr_matrix["median_house_value"]

'''
Another way to check for correlation between attributes is to use the pandas scatter_matrix,
which plot every numerical attribute again each other.
Since we have 11 atribut, we will get 11*11 = 121 plot, which is too much. 
Let focus on some promising attribue that seem most correlated with median house value.
'''

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


'''
we can see that the most promising attribute to predict median_house value is 
the median_income (strong upward trend). If we zoom at it, we see that the plot
reveal some not obvious straight lines: around 500k, 450k, 350k.
We may try to remove the corresponding districts to prevent our algorithms from
learning to reproduce this data quirks.
'''
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)







