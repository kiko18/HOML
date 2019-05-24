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
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)  *test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    