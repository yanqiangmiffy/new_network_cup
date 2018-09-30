# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: utils.py 
@Time: 2018/9/30 13:07
@Software: PyCharm 
@Description:
"""
import pandas as pd


def load_data():
    df_train=pd.read_csv('input/train_xy.csv') # 含有y
    df_test=pd.read_csv('input/test_all.csv') # 不含有y
    # 去除异常值(-99)
    n_row = len(df_train)
    for col in df_train.columns:
        cnt = (df_train[col] == -99).astype(int).sum()
        if (float(cnt) / n_row) > 0.45:
            df_train.drop([col], axis=1, inplace=True)
            df_test.drop([col], axis=1, inplace=True)
    print(df_train.shape,df_test.shape)
    return df_train,df_test