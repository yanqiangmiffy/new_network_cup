# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 07_ensemble_stacking.py
@Time: 2018/10/16 18:07
@Software: PyCharm 
@Description:
"""

import pandas as pd
import numpy as np
from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# 设置随机种子
SEED=222
np.random.seed(SEED)
df_train,df_test=load_data(filter_flag=True,process_flag=False)


def get_train_test(test_size=0.2):
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=get_train_test()
