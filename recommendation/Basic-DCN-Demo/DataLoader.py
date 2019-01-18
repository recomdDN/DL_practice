import numpy as np
import pandas as pd

# 为类别型特征生成字典{'col_name':{value1: order1, value2: order2,...},...}
class FeatureDictionary(object):
    def __init__(self, dfTrain=None, dfTest=None,
                 numeric_cols=[],
                 ignore_cols=[],
                 cate_cols=[]):

        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.cate_cols = cate_cols
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.dfTrain, self.dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            # 不处理忽略的特征和数值型特征
            if col in self.ignore_cols or col in self.numeric_cols:
                continue
            # 处理类别型特征和其余特征
            # {'col_name':{value1: order1, value2: order2,...},...}
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        # 总离散特征个数
        self.feat_dim = tc

#
class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    # 将原数据dataframe解析为(类别型特征id, 类别型特征, 数值型特征, 标签值)
    def parse(self, df=None, has_label=False):
        assert df is not None, "parameter df should be specified"
        dfi = df.copy()
        y = []
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["target"], axis=1, inplace=True)
        dfi.drop(["id"], axis=1, inplace=True)

        # 数值型特征
        numeric_Xv = dfi[self.feat_dict.numeric_cols].values.tolist()
        dfi.drop(self.feat_dict.numeric_cols, axis=1, inplace=True)


        dfv = dfi.copy()
        for col in dfi.columns:
            # 删除忽略的特征
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            # 处理类别型特征
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                # 同一个类别型特征取不同值时权重都为1
                dfv[col] = 1.

        # 类别型特征索引
        cate_Xi = dfi.values.tolist()
        # 类别型特征值
        cate_Xv = dfv.values.tolist()
        if has_label:
            return cate_Xi, cate_Xv, numeric_Xv, y
        else:
            return cate_Xi, cate_Xv, numeric_Xv, y
