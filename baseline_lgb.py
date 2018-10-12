import os
base_path='input/'
os.chdir(base_path)


import pandas as pd
import numpy as np
train_withlabel=pd.read_csv('train_xy.csv')
train_nolabel=pd.read_csv('train_x.csv')
test = pd.read_csv('test_all.csv')
x_test = test
x = train_withlabel.drop(['y','cust_id','cust_group'],axis=1)
x_test=x_test.drop(['cust_id','cust_group'],axis=1)
y = train_withlabel['y']


from sklearn.model_selection import train_test_split
#x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=1230)
x_train = x
y_train = y
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from lightgbm import LGBMClassifier
clf = LGBMClassifier(n_jobs=-1,
                     n_estimators=200,
                     learning_rate=0.01,
                     num_leaves=34,
                     colsample_bytree=0.9,
                     subsample=0.9,
                     max_depth=8,
                     reg_alpha=0.04,
                     reg_lambda=0.07,
                     min_split_gain=0.02,
                     min_child_weight=40,
                    )
clf.fit(x_train,y_train)
lgb_pred=clf.predict_proba(x_test)[:,1]

lgb_sub = pd.DataFrame({'cust_id':test['cust_id'],'y_pred':lgb_pred})
lgb_sub.to_csv('lgb_sub.csv',index=0)


