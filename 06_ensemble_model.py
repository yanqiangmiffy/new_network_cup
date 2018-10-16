# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 03_sklearn_base_demo.py
@Time: 2018/9/29 17:48
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
df_train,df_test=load_data(filter_flag=True,process_flag=True)


def ensemble_demo():
    lr_result=pd.read_csv('result/01_sklearn_lr.csv')
    lgb_result=pd.read_csv('result/02_lgb.csv')

    pred_prob=lgb_result.pred_prob+lr_result.pred_prob
    print(pred_prob/2)

    df_test['pred_prob'] = pred_prob/2
    df_test[['cust_id', 'pred_prob']].to_csv('result/06_ensemeble.csv', index=False)


def get_train_test(test_size=0.2):
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test=get_train_test()


def get_models():
    """Generate a library of base learners."""
    lr = LogisticRegression(C=4.0, random_state=SEED)
    # rf = RandomForestClassifier(random_state=SEED)
    # gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    # ab = AdaBoostClassifier(random_state=SEED)
    xgb = XGBClassifier()
    lgb = LGBMClassifier()
    models = {'logistic': lr,
              # 'random forest': rf,
              # 'gbm': gb,
              # 'ab': ab,
              'xgb': xgb,
              'lgb': lgb
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_test.shape[0], len(model_list)))

    x_sub = df_test.drop(['cust_id', 'cust_group'], axis=1, inplace=False)
    P_sub = np.zeros((x_sub.shape[0], len(model_list)))

    P = pd.DataFrame(P)
    P_sub = pd.DataFrame(P_sub)

    print("Fitting models.")

    cols = list()

    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)

        m.fit(X_train, y_train)

        P.iloc[:, i] = m.predict_proba(X_test)[:, 1]
        P_sub.iloc[:, i] = m.predict_proba(x_sub)[:, 1]

        cols.append(name)

        print("done")

    P.columns = cols
    P_sub.columns = cols

    print("Done.\n")

    return P,P_sub


def score_models(P, y):
    """Score model in prediction DF"""

    print("Scoring models.")

    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])

        print("%-26s: %.3f" % (m, score))

    print("Done.\n")


def predict(P_sub):
    df_test['pred_prob'] = P_sub.mean(axis=1)
    df_test[['cust_id', 'pred_prob']].to_csv('result/06_lgb_all.csv', index=False)
    print("predictin done")

models = get_models()
P,P_sub = train_predict(models)
score_models(P, y_test)
predict(P_sub)