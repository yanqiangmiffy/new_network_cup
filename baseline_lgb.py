from utils import load_data
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
df_train,df_test=load_data(filter_flag=True,process_flag=True)


def train():
    X = df_train.drop(['cust_id', 'y', 'cust_group'], axis=1, inplace=False)
    y = df_train['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    print("roc_auc_score：", roc_auc_score(y_test, prob[:, 1]))
    return clf


gbm=train()


# 预测提交
def predict():
    eval_x = df_test.drop(['cust_id', 'cust_group'], axis=1, inplace=False)
    print('Start predicting...')
    eval_pred = gbm.predict_proba(eval_x)
    df_test['pred_prob'] = eval_pred[:, 1]
    df_test[['cust_id', 'pred_prob']].to_csv('result/04_baseline_lgb.csv', index=False)


predict()









