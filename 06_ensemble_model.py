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
from utils import load_data
df_train,df_test=load_data(filter_flag=True,process_flag=True)

lr_result=pd.read_csv('result/01_sklearn_lr.csv')
lgb_result=pd.read_csv('result/02_lgb.csv')

pred_prob=lgb_result.pred_prob+lr_result.pred_prob
print(pred_prob)

df_test['pred_prob'] = pred_prob
df_test[['cust_id', 'pred_prob']].to_csv('result/06_ensemeble.csv', index=False)