# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#######################################################
#                     数据准备
#######################################################

# 首先将csv中的PM2.5数据提取出来，保存在pm2_5这个dataframe中
file = open("C:\\Users\\lidan\\Desktop\\train.csv", encoding='UTF-8')
trainSet = pd.read_csv(file)
pm2_5 = trainSet[trainSet["item"] == "PM2.5"]
pm2_5.drop(["date", "station", "item"], axis=1, inplace=True)

# 将原始形式的PM2.5数据保存为训练数据x及标签y
x = []
y = []

for i in range(15):
    tempx = pm2_5.iloc[:, i:i+9];
    tempx.columns = np.array(range(9))
    x.append(tempx)
    tempy = pm2_5.iloc[:, i+9]
    tempy.columns = ["1"]
    y.append(tempy)
    
x = pd.concat(x)
y = pd.concat(y)  #3600x1 ,y为serise
x = np.array(x, float)
y = np.array(y, float)

#######################################################
# 用梯度下降训练模型
#######################################################
# adding baias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
# 初始化一个参数矩阵
w=np.zeros((len(x[0])))

#初始化learning rate
lr = 10
iteration = 10000
s_grad = np.zeros(len(x[0]))
for i in range(iteration):
    tem = np.dot(x,w)     # y预测值
    loss = y-tem     
    grad = np.dot(x.transpose(),loss)*(-2)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - lr*grad/ada  # 跳出循环得到训练完毕的参数向量w
    
#######################################################
# 将模型应用到测试数据集
#######################################################
testdata = pd.read_csv("C:\\Users\\lidan\\Desktop\\test.csv")
# 同样地只取出来PM2.5那一行的数据作为feature
feature = testdata[testdata["item"] == "PM2.5"]
feature.drop(["id", "item"], axis=1, inplace=True)
feature = np.array(feature, float)
# adding bias
ones = np.ones((feature.shape[0], 1))
feature = np.concatenate((ones, feature), axis=1)
# 用模型跑测试数据
y_final = np.dot(feature, w)
# 计算误差
y_for = pd.read_csv("C:\\Users\\lidan\\Desktop\\sampleSubmission.csv")
y_for.value = y_final
y_real = pd.read_csv("https://ntumlta.github.io/2017fall-ml-hw1/ans.csv")
err = abs(y_for.value-y_real.value).sum()/len(y_real.value)
print(err)