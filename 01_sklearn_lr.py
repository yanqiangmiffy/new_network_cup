# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:yanqiang
@File: 03_sklearn_base_demo.py
@Time: 2018/9/29 17:48
@Software: PyCharm
@Description:
"""
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,mean_squared_error,log_loss
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from utils import load_data

df_train,df_test=load_data()


# 调整参数
def tune_params(X,y):
    param_test1={'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']} # 'newton-cg'
    # param_test2 = {'C': [1.0, 3.0, 4.0,5.0]} # 5.0 最佳
    # param_test3 = {'max_iter': range(0, 200, 20)}  # 100最佳
    # param_test4={'multi_class':['ovr', 'multinomial']} # multinomial
    gsearch = GridSearchCV(estimator=LogisticRegression(C=5.0,max_iter=20,random_state=10),
                            param_grid=param_test1, scoring='roc_auc')
    gsearch.fit(X, y)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


def extract_feature(X, y):
    from sklearn.feature_selection import VarianceThreshold
    #设置方差的阈值为0.8
    sel = VarianceThreshold(threshold=.08)
    #选择方差大于0.8的特征
    sel.fit(X)
    return X,sel


# 预测提交
def predict(clf,pipeline):
    eval_x = df_test.drop(['cust_id', 'cust_group'], axis=1, inplace=False)
    # eval_x = pipeline.fit_transform(eval_x)

    submit_pred = clf.predict_proba(eval_x)
    submit_pred = submit_pred[:, 1]  # 风险高的用户概率
    df_test['pred_prob'] = submit_pred
    df_test[['cust_id', 'pred_prob']].to_csv('result/01_sklearn_lr.csv', index=False)


def main():
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train,X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', MinMaxScaler()),
    ])
    # X_train=pipeline.fit_transform(X_train)
    # X_test=pipeline.fit_transform(X_test)

    print(X_train.shape, X_test.shape)

    # X_train=extract_feature(X_train,y_train)
    clf=LogisticRegression(C=1.0,max_iter=100,random_state=10)

    print("===="*20)
    clf.fit(X_train, y_train)
    prob  = clf.predict_proba(X_test)
    pred = np.argmax(prob, axis=1)
    print("mean_squared_error:", mean_squared_error(y_test, prob[:, 1]))
    print("log_loss:", log_loss(y_test.astype(int), prob[:, 1]))
    print("roc_auc_score：", roc_auc_score(y_test, prob[:, 1]))
    # high_danger_prob=prob[:, 1]
    # print(high_danger_prob)

    # print("调参")
    # tune_params(X_test, y_test)

    predict(clf,pipeline)

if __name__ == '__main__':
    main()




