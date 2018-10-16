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
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import Imputer

def load_data(filter_flag=False,process_flag=False):
    """
    加载数据，是否需要去除异常值
    :param filter_flag: True：去除异常值
    :return:
    """
    df_train=pd.read_csv('input/train_xy.csv') # 含有y
    df_test=pd.read_csv('input/test_all.csv') # 不含有y
    if filter_flag: # 是否去除异常值(-99)
        n_row = len(df_train)
        for col in df_train.columns:
            cnt = (df_train[col] == -99).astype(int).sum()
            if (float(cnt) / n_row) > 0.9:
                df_train.drop([col], axis=1, inplace=True)
                df_test.drop([col], axis=1, inplace=True)
    print("df_train、df_test原有数据维度:",df_train.shape, df_test.shape)

    if process_flag: # 是否进行预处理
        # 类别特征处理发现有问题：数据类别不一致，这里需要舍弃几列
        drop_label=['x_139','x_147','x_151','x_152','x_153','x_154','x_155']
        df_train.drop(drop_label, axis=1, inplace=True)
        df_test.drop(drop_label, axis=1, inplace=True)

        # 特征变量x1-x95是数值型变量，x96-x157是类别型变量
        scaler = StandardScaler()

        # 类别特征处理
        label_cols=list(df_train.loc[:,'x_97':'x_157'].columns)
        # label_encoder=LabelEncoder()
        # df_train[label_cols]=df_train[label_cols].apply(label_encoder.fit_transform)
        # df_test[label_cols]=df_train[label_cols].apply(label_encoder.fit_transform)

        df_train=pd.get_dummies(df_train,columns=label_cols)
        df_test=pd.get_dummies(df_test,columns=label_cols)
        print(set(list(df_train.columns))-set(list(df_test.columns)))
        print(set(list(df_test.columns))-set(list(df_train.columns)))

        # 数值特征处理
        num_cols=list(df_train.loc[:,'x_1':'x_96'].columns)
        df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
        df_test[num_cols] = scaler.transform(df_test[num_cols])

        # 异常值处理 Imputer-median 用特征列的中位数替换
        # imputer=Imputer(missing_values=-99,strategy='median')
        # df_train[num_cols]=imputer.fit_transform(df_train[num_cols])
        # df_test[num_cols]=imputer.transform(df_test[num_cols])
        print("df_train、df_test数据处理后数据维度:",df_train.shape, df_test.shape)

    return df_train,df_test


import seaborn

