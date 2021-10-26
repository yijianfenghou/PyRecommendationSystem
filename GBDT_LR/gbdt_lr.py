import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('DataSet/train.csv')
df_test = pd.read_csv('DataSet/test.csv')

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

y_train = df_train['target']
y_test = df_test['target']
X_train = df_train[NUMERIC_COLS]
X_test = df_test[NUMERIC_COLS]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves, will be used in feature transformation
num_leaf = 64

# train
gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_train)

gbm.save_model('model.txt')

y_pred = gbm.predict(X_train, pred_leaf=True)

print(y_train)
print(np.array(X_train).shape)
print(np.array(y_pred).shape)

transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0])*num_leaf], dtype=np.int64)

for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][tmp] += 1

y_pred = gbm.predict(X_test, pred_leaf=True)
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

lm = LogisticRegression(penalty='l2', C=0.5)
lm.fit(transformed_training_matrix, y_train)
y_pred_test = lm.predict_proba(transformed_testing_matrix)

print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
print("Normalized Cross Entropy " + str(NE))