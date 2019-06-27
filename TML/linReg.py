# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:09:15 2019

@author: basil
"""

import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

# get root directory and add it to the system path
import os
import sys
currDir = os.getcwd()
rootDir = os.path.abspath(os.path.join(currDir, os.pardir))
sys.path.append(rootDir)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(rootDir, "data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

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
plt.show()

'''
Now we will investigates the correlation  between attributes.
We can compute the standart correlation coefficient (also called pearson's r)
between every pair of attribues using the corr() methods)
Note that the correlation coef only measure linear correlation (if x goes up, y goes up/down)
'''
corr_matrix = housing.corr()

#we can for example look at how much each attribute correlates with the median house value.
#The correlation coef range from -1 to 1.  1 indicate a strong positive corelation.
#for example, the median house tend to go up when the median income goes up
#-1 indicates a strong negative correlation. For example between the latitude 
#and the median house value (i.e prices have a slight tendency to goes down when you go north)
#Finally, coef close to zero means that there is no linear corelation
corr_matrix["median_house_value"].sort_values(ascending=False)

'''
Another way to check for correlation between attributes is to use the pandas scatter_matrix,
which plot every numerical attribute again each other.
Since we have 11 atribut, we will get 11*11 = 121 plot, which is too much. 
Let focus on some promising attribue that seem most correlated with median house value.
'''

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


'''
we can see that the most promising attribute to predict median_house value is 
the median_income (strong upward trend). If we zoom at it, we see that the plot
reveal some not obvious straight lines: around 500k, 450k, 350k.
We may try to remove the corresponding districts to prevent our algorithms from
learning to reproduce this data quirks.
'''
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

'''
until now we have see few way to explore the data and get insight from it.
We identified a few quirks that we might want to clean before feeding the data to a ML algo.
We found interessting correlation between attribute, in particular with the target attribute.
We notice that some attribute have a tail-heavy distribution, so we might want to transform them
for example by computing the logarithms.
Of course our maileage will vary considerably within each project but the general idea are similar.

One last thing we might want to do before actually preparing the data for a ML algo is to try out
various attribute combinations. For example the total number of bedroom in a distric is not very
useful is we don't know how many household they are. want we really want is the number of room per 
household.
SImilary the number of bedroom by itself is not very usefull, what we want is to compare it with the
number of room. The population per household also seem like an interesting atribute.
'''

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#let's look at the correlation again
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

'''
Cool, we see that the new roms_per_household is more correlated with the median_house_value than the
total number of rooms or bedrooms. Aparently, houses with a lower bedroom/room ratio tend to be more 
expensive.

We stop now with the data exploration for now. Generally, we do quick round of data exploration to 
get insight that will help us to build a first reasonable good model. Once we have prototype up and
runing, we can analyze it output to gainmore insights and come back to this exploration step
'''


'''
Now we will prepare the data for Machine learning algorithms.
First we will separate the predictors and the labels  
'''
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

'''
Data cleaning:
Most machine learning algo cannot work with missing features.
We noticied earlier that the total_bedrooms attribute has some missing values.
(for example for the districs 2412, 6220, ...) 
to fix this, we have 3 options: 
    - get ride of the corresponding districs
          housing.dropna(subset=["total_bedrooms])
    - get rid of the whole attribute/feature
          housing.drop("total_bedrooms", axis=1)  
    - set the values to some value (zero, the mean, the median, etc)
          median = housing["total_bedrooms"].median()
          housing["total_bedrooms"].fillna(median, inplace=true)
          when you choose this option, don't forget to save the median, you will use it 
          later to replace missing values in the test set when you want to evaluate the system
          and also once the system goes live to replace missing value in new data
'''

'''
sklearn provides a class "SimpleImputer" to takes care of missing values
'''
from sklearn.impute import SimpleImputer
# create an instance specifying that we want to replace each atribute missing values 
# with the median of that attribute
imputer = SimpleImputer(strategy="median") 
# since the median can only be computed on numerical atribute, we need to create a copy
# of the data without the text attribute ocean_proximity
housing_temp = housing.drop("ocean_proximity", axis=1)
# now we can fit the imputer instance to the training data. 
# it compute the median of each attribute and store the result in its statistics_ instance variable.
# In fact, only the total_bedrooms attribute has missing values, but we cannot be sure that there
# won't be any missing values in new data after the system goes live, so it is sater to apply
# the imputer to all the numerical attributes
imputer.fit(housing_temp)
print(imputer.statistics_)
# now we can use this "trained" imputer to transform the training set by replacing missing values
# by the learned medians. The result is a plain Numpy array containing the transformed features.
# If we want we can put it back into a pandas DataFrame 
X = imputer.transform(housing_temp) #there is also a method fit_transform wich do both fit ans transform
housing_tr = pd.DataFrame(X, columns=housing_temp.columns)

# Earlier we left out the categorical attribute "ocean_proximity" becuase it is a text attribute
# so we can not compute its median.
# Most ML algo prefer to work with numbers anyway, so let's convert these categories from text to numbers
from sklearn.preprocessing import OrdinalEncoder
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) 
print(ordinal_encoder.categories_)  #get the list of categories

'''
one issue with this housing_cat_encoded representation is that ML algo will assume that 2 nearby values
are more similar than 2 distant values. This may be fine in some cases (eg. for ordered categories such 
as bad, average, good, excellent) but is is obviously not the case for ocean proximety(for example,
categorie 0 and 4 are clearly more similar than categories 0 and 1).

To fix this issue, a common solution is to create one binary attribute per category. i.e for each 
training example we will have 4 column value where only one of them is 1.
This is called a one-hot encoding. namely because only one attribut is equal to one.
'''
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder() #create an instance
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) #train the instance
# note that the result is a Scipy sparse matrix, instead of a NumPy array
# this allow to save memory in case you have a lot of categories. 
print(housing_cat_1hot) 
# in our case Scipy matrix is not too large, we can convert it to a numpy array
housing_cat_1hot.toarray()

# test123456




