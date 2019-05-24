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
Create a datasets/housing directory in your workspace, 
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
    test_set_size = int(len(data)  *test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


'''
Supose we have chatted with experts who told us that the median income is a very important attribute 
to predict median housing price. We may want to ensure that the test set is representative of the various
categories of incomes in the whole datasets. Since the median income is a continous numerical attribute, 
we first need to create an income category attribute.
'''
#create an incoming category attribute
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


#Do stratified sampling 
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
'''
To see if this worked as expected we can look at the income category proportions 
in the full housing dataset. This should almost identical to the income category proportions 
in the training and test set
'''
print(housing["income_cat"].value_counts() / len(housing))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

'''
If we check these proportion for the test set generated via purely random sampling,
we will see that it is quite skewed
'''
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(housing["income_cat"].value_counts() / len(housing))
print(train_set["income_cat"].value_counts() / len(train_set))
print(test_set["income_cat"].value_counts() / len(test_set))

#we remove the income_cat so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)