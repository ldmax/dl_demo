# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

"""
Created on Thu Feb 22 21:10:36 2018

@author: lidan
"""
##################################
#    data preparation
##################################

# reading data
dataframe = pd.read_csv("../train.csv")
label = dataframe["income"]
dataframe.drop(["income"], axis=1, inplace=True)

# convert categorical data into numeric data
dataframe["workclass"] = dataframe["workclass"].astype("category")
dataframe["marital_status"] = dataframe["marital_status"].astype("category")
dataframe["occupation"] = dataframe["occupation"].astype("category")
dataframe["relationship"] = dataframe["relationship"].astype("category")
dataframe["race"] = dataframe["race"].astype("category")
dataframe["sex"] = dataframe["sex"].astype("category")
dataframe["native_country"] = dataframe["native_country"].astype("category")
cat_columns = dataframe.select_dtypes(["category"]).columns
dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes) # x: a Series in dataframe
label = label.astype("category")
label = label.cat.codes
# drop column "education"
dataframe.drop(["education"], axis=1, inplace=True)
# convert DataFrame to ndarray so that matrix manipulation can be conveyed
train = dataframe.as_matrix()
label = label.as_matrix()
##################################
#    logistic regression
##################################

