# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 03_sklearn_base_demo.py
@Time: 2018/9/29 17:48
@Software: PyCharm 
@Description:
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error,log_loss
import pandas as pd
import numpy as np
from utils import load_data

df_train,df_test=load_data(filter_flag=True)


def train():
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)

    names = [
        "KNeighborsClassifier",
        "SGDClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "AdaBoostClassifier",
        "DecisionTreeClassifier"]
    classifiers = [
        KNeighborsClassifier(),
        SGDClassifier(loss='log'),
        LogisticRegression(C=4.0),
        RandomForestClassifier(oob_score=True),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        DecisionTreeClassifier()]
    for name,clf in zip(names,classifiers):
        print("===="*20)
        print("traing..."+name)
        clf.fit(X_train, y_train)

        prob  = clf.predict_proba(X_test).astype(float)
        print(prob)
        # pred = np.argmax(prob, axis=1)
        print("mean_squared_error:",mean_squared_error(y_test,prob[:,1]))
        print("log_loss:",log_loss(y_test.astype(int),prob[:,1]))
        print("roc_auc_score：",roc_auc_score(y_test,prob[:,1]))
        # high_danger_prob=prob[:, 1]
        # print(high_danger_prob)


train()
