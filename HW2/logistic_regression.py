# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn import preprocessing

"""
Created on Thu Feb 22 21:10:36 2018

@author: lidan
"""
##################################
#    data preparation
##################################


def data_prepare(path):
    """
        从path读取csv数据，将categorical数据转为连续的整数，然后用OneHotEncoder
        再编码。将连续型数据用除以列最大值的方式scaling，然后合并连续型和categorical
        数据。
    Args:    
        path 读取csv数据的路径，可以是本地路径或网址
    Returns:
        data_cat DataFrame 返回的categorical数据
        dataframe DataFrame 返回的连续型数据
        y Series 返回的label数据
    """
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
    dataframe["education"] = dataframe["education"].astype("category")
    if len(dataframe.columns) == 15:
        dataframe["income"] = dataframe["income"].astype("category")
    
    cat_columns = dataframe.select_dtypes(["category"]).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)  # x: a Series in dataframe
    data_cat = dataframe[cat_columns]
    dataframe.drop(cat_columns, axis=1, inplace=True)
    # drop column "education_num"
    dataframe.drop(["education_num"], axis=1, inplace=True)
    y = np.zeros((dataframe.shape[0], 1))
    if len(data_cat.columns) == 9:
        y = data_cat["income"].as_matrix()
        data_cat.drop(["income"], axis=1, inplace=True)
    
    return data_cat, dataframe, y


data_cat_train, data_cont_train, y_train = data_prepare("D:/lihongyi/dl/classification/dl_demo/HW2/train.csv")
data_cat_test, data_cont_test, y_test = data_prepare("D:/lihongyi/dl/classification/dl_demo/HW2/test.csv")
data_cat = pd.concat([data_cat_train, data_cat_test], axis=0)
# use OneHotEncoder to fit merged categorical data
encoder = preprocessing.OneHotEncoder()
encoder.fit(np.array(data_cat))
x_cat_train = encoder.transform(data_cat_train).toarray()
x_cat_test = encoder.transform(data_cat_test).toarray()

# before concatenating, normalize continuous data
for column in data_cont_train:
    data_cont_train[column] = data_cont_train[column] / data_cont_train[column].max()
    data_cont_test[column] = data_cont_test[column] / data_cont_test[column].max()

# concatenate categorical data and continuous data
x_cont_train = data_cont_train.as_matrix()
x_cont_test = data_cont_test.as_matrix()
x_train = np.hstack((x_cont_train, x_cat_train))
x_test = np.hstack((x_cont_test, x_cat_test))
##################################
#    logistic regression
##################################


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def gradient_decent(lr, iteration, x, y):
    """
    梯度下降
    Args:
        lr: int learning rate初始值
        iteration: int 迭代次数
        x: ndarray 训练数据矩阵
        y: ndarray 标签数组
    Returns:
        w: 训练完成的参数向量
    """
    w = np.zeros((len(x[0])))  # parameter vector initialization
    s_grad = np.zeros(len(x[0]))
    
    for i in range(iteration):
        z = np.dot(x, w)
        temp = []
        for j in z.flat:
            temp.append(sigmoid(j))  # Sigmoid function
    
        loss = temp - y
        grad = np.dot(x.transpose(), loss)
        s_grad += grad**2
        ada = np.sqrt(s_grad)
        w = w - (lr/ada) * grad
    return w

##################################
#    model training
##################################


x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)  # adding bias
lr = 10  # learning rate initialization
iteration = 10000
w = gradient_decent(lr, iteration, x_train, y_train)
##################################
#       test
##################################
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)  # adding bias
z_test = np.dot(x_test, w)
income = []
for i in z_test.flat:
    income.append(sigmoid(i))

income = np.array(income)
# error calculation

y_for = pd.read_csv("D:/lihongyi/dl/classification/dl_demo/HW2/correct_answer.csv")
y_for.label = income
y_for["label"] = y_for["label"].apply(lambda x: 1 if x > 0.5 else 0)

y_real = pd.read_csv("D:/lihongyi/dl/classification/dl_demo/HW2/correct_answer.csv")


def f1_measure(ans, pre):
    """
    计算分类问题的F1-measure
    Args:
        ans ndarray 正确答案
        pre ndarray 预测
    Returns:
        accuracy float 准确率
    """
    if len(ans) == len(pre):
        length = len(ans)
        tp = 0
        fp = 0
        fn = 0
        for i in range(length):
            # 计算True Positive的数量
            if ans[i] == 1 and pre[i] == 1:
                tp += 1
            # 计算False Positive的数量
            elif ans[i] == 0 and pre[i] == 1:
                fp += 1
            # 计算False Negative的数量
            elif ans[i] == 1 and pre[i] == 0:
                fn += 1
        
        p = tp / (tp + fp)  # precision
        r = tp / (tp + fn)  # recall

        return 2*p*r / (p + r)  # F1-measure
            

f1 = f1_measure(np.array(y_real.label), np.array(y_for.label))
print(f1)

