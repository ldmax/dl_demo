# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math

"""
Created on Thu Feb 22 21:10:36 2018

@author: lidan
"""

##################################
#    data preparation
##################################


def data_prepare(path):
    # reading data
    dataframe = pd.read_csv(path)
    
    # convert categorical data into numeric data
    dataframe["workclass"] = dataframe["workclass"].astype("category")
    dataframe["marital_status"] = dataframe["marital_status"].astype("category")
    dataframe["occupation"] = dataframe["occupation"].astype("category")
    dataframe["relationship"] = dataframe["relationship"].astype("category")
    dataframe["race"] = dataframe["race"].astype("category")
    dataframe["sex"] = dataframe["sex"].astype("category")
    dataframe["native_country"] = dataframe["native_country"].astype("category")
    if len(dataframe.columns) == 15:
        dataframe["income"] = dataframe["income"].astype("category")

    cat_columns = dataframe.select_dtypes(["category"]).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)  # x: a Series in dataframe
    # drop column "education"
    dataframe.drop(["education"], axis=1, inplace=True)
    
    y = np.zeros((dataframe.shape[0], 1))
    if len(dataframe.columns) == 14:
        y = dataframe["income"].as_matrix()
        dataframe.drop(["income"], axis=1, inplace=True)
    
    # scaling and normalization for multi-feature data set
    # scaling: divided by max value of the column
    for column in dataframe.columns:
        dataframe[column] = dataframe[column] / dataframe[column].max()
    
    # normalization: each column minus its mean value then divided by its standard variation
    # for column in dataframe.columns:
    #     dataframe[column] = (dataframe[column] - dataframe[column].mean()) / dataframe[column].std()
    
    # convert DataFrame to ndarray so that matrix manipulation can be conveyed
    x = dataframe.as_matrix()
    return x, y
##################################
#    logistic HW1
##################################

# using adagrad to adapt learning rate


def gradient_decent(lr, iteration, x, y):
    w = np.zeros((len(x[0])))  # parameter vector initialization
    s_grad = np.zeros(len(x[0]))
    
    for i in range(iteration):
        z = np.dot(x, w)
        temp = []
        for j in z.flat:
            temp.append(1/(1+math.exp(j)))  # Sigmoid function
    
        loss = temp - y
        grad = np.dot(x.transpose(), loss)*(-2)
        s_grad += grad**2
        ada = np.sqrt(s_grad)
        w = w - (lr/ada) * grad
    return w

##################################
#    model training
##################################


x, y = data_prepare("https://ntumlta.github.io/2017fall-ml-hw2/raw_data/train.csv")
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)  # adding bias
lr = 10  # learning rate initialization
iteration = 10000
w = gradient_decent(lr, iteration, x, y)

##################################
#       test
##################################


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


x_test, y_test = data_prepare("https://ntumlta.github.io/2017fall-ml-hw2/raw_data/test.csv")
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)  # adding bias
z_test = np.dot(x_test, w)
income = []
for i in z_test.flat:
    income.append(sigmoid(i))

income = np.array(income)
# error calculation

y_for = pd.read_csv("https://ntumlta.github.io/2017fall-ml-hw2/correct_answer.csv")
y_for.label = income
y_for["label"] = y_for["label"].apply(lambda x: 1 if x > 0.9 else 0)

y_real = pd.read_csv("https://ntumlta.github.io/2017fall-ml-hw2/correct_answer.csv")
err = abs(y_for.label-y_real.label).sum()/len(y_real.label)
print(err)

