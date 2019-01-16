import lightgbm as lgb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


print('Load data...')
data_path = '../Basic-AFM-Demo/'

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

TARGET = ["target"]
data = pd.read_csv(data_path + 'data/train.csv', usecols=NUMERIC_COLS + TARGET)

X, y = data[NUMERIC_COLS], data[TARGET].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

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

# number of leaves,will be used in feature transformation
num_leaf = 64

print('开始训练模型...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('保存模型...')
# save model to file
gbm.save_model('model.txt')

print('开始预测...')

# shape = [num_sample, num_trees], y_pred存在的是leaf_index
y_pred = gbm.predict(X_train, pred_leaf=True)
print('对训练数据进行转换')
# shape = [num_sample, num_trees * num_leafs], 存放的是0,1
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(X_test, pred_leaf=True)
print('对测试数据进行转换')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

lm = LogisticRegression(penalty='l2', C=0.05)  # 逻辑回归
lm.fit(transformed_training_matrix, y_train)   # 训练转换后的训练数据
y_pred_test = lm.predict_proba(transformed_testing_matrix)  # 预测转换后的测试数据

print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(
    ((1 + y_test) / 2 * np.log(y_pred_test[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_test[:, 1])))
print("Normalized Cross Entropy " + str(NE))
