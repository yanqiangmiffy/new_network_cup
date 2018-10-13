# -*- coding: utf-8 -*-
# @Time    : 2018/10/13 22:54
# @Author  : quincyqiang
# @File    : 05_smote_xgb.py
# @Software: PyCharm

from xgboost import XGBClassifier
from utils import load_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

df_train,df_test=load_data(filter_flag=True,process_flag=False)
X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
y = df_train['y']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25, random_state = 42)
xgb=XGBClassifier(random_state=42,n_jobs=-1)
xgb.fit(X_train, y_train)
#Make predictions
print('Classification of SMOTE-resampled dataset with XGboost')
# y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)
print(roc_auc_score(y_test,y_prob[:, 1]))
if __name__ == '__main__':
    pass