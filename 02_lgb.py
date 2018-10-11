# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 02_lgb.py
@Time: 2018/9/29 17:47
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from utils import load_data
from sklearn.model_selection import train_test_split

df_train,df_test=load_data()

def train():
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'auc'},
        # 'is_unbalance': True,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    return gbm

gbm=train()


# 预测提交
def predict():
    eval_x = df_test.drop(['cust_id', 'cust_group'], axis=1, inplace=False)
    print('Start predicting...')
    y_pred = gbm.predict(eval_x, num_iteration=gbm.best_iteration)
    df_test['pred_prob'] = y_pred
    df_test[['cust_id', 'pred_prob']].to_csv('result/submit.csv', index=False)

predict()