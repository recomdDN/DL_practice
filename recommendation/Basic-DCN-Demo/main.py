import tensorflow as tf

import pandas as pd
import numpy as np

import config

from sklearn.model_selection import StratifiedKFold
from DataLoader import FeatureDictionary, DataParser

from DCN import DCN


def load_data():
    # 读取数据
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)


    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        # 除了id, target列外，每一个数据缺失特征数
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        # 手动交叉特征
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)
    return dfTrain, dfTest


def run_base_model_dcn(dfTrain, dfTest, folds, dcn_params):
    # 类别型特征与索引的映射
    fd = FeatureDictionary(dfTrain, dfTest, numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           cate_cols=config.CATEGORICAL_COLS)

    print(fd.feat_dim)
    print(fd.feat_dict)

    # 返回类别型特征索引，类别型特征值，数值型特征，标签值
    data_parser = DataParser(feat_dict=fd)
    cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    cate_Xi_test, cate_Xv_test, numeric_Xv_test, _ = data_parser.parse(df=dfTest)

    # 离散型特征onthot后类别型特征个数
    dcn_params["n_cate_feature"] = fd.feat_dim
    # 离散型特征个数
    dcn_params["n_field"] = len(cate_Xi_train[0])
    print('values', str(fd.feat_dim), 'values', str(len(cate_Xi_train[0])))
#
    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        # 训练集
        cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_, y_train_ = _get(cate_Xi_train, train_idx), _get(
            cate_Xv_train, train_idx), _get(numeric_Xv_train, train_idx), _get(y_train, train_idx)
        # 验证集
        cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_, y_valid_ = _get(cate_Xi_train, valid_idx), _get(
            cate_Xv_train, valid_idx), _get(numeric_Xv_train, valid_idx), _get(y_train, valid_idx)

        dcn = DCN(**dcn_params)

        dcn.fit(cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_, y_train_, cate_Xi_valid_, cate_Xv_valid_,
                numeric_Xv_valid_, y_valid_)


dfTrain, dfTest = load_data()
print('load_data_over')

# 分层采样交叉验证，会根据标签值来保证训练集和测试集的样本分别相同
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(np.zeros(len(dfTrain)), dfTrain['target']))
print('process_data_over')

dcn_params = {
    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "random_seed": config.RANDOM_SEED,
    "cross_layer_num": 3,
    "n_numeric_feature": config.N_NUMERIC_FEATURE
}
print('start train')
run_base_model_dcn(dfTrain, dfTest, folds, dcn_params)
