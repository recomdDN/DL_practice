import numpy as np
import pandas as pd
import tensorflow as tf
import os

DATA_DIR = 'data/ratings.dat'
DATA_PATH = 'data/'
COLUMN_NAMES = ['user', 'item', 'timestamp']

# 编号映射 s-->order, 将原来的编号重新从0开始编号
def re_index(s):
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map


def load_data():
    # 只读取user, item列
    full_data = pd.read_csv(DATA_DIR, sep='::', header=None, names=COLUMN_NAMES,
                            usecols=[0, 1, 3], dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine='python')
    # 将user从0开始编号
    full_data['user'] = full_data['user'] - 1
    # 用户编号集合
    user_set = set(full_data['user'].unique())
    # 用户个数
    n_user = len(user_set)

    # item编号字典
    item_set = set(full_data['item'].unique())
    item_map = re_index(item_set)
    # 将item原有的编号转换成新的从0开始的编号
    full_data['item'] = full_data['item'].map(lambda x: item_map[x])

    # 商品编号集合
    item_set = set(full_data['item'].unique())
    # 商品个数
    n_item = len(item_set)

    # user_id-->item_id_list
    # 对应这用户已购买的清单
    user_bought = {}
    for i in range(len(full_data)):
        u = full_data['user'][i]
        t = full_data['item'][i]
        if u not in user_bought:
            user_bought[u] = []
        user_bought[u].append(t)

    # user_id-->item_id_list
    # 对应这用户未购清单
    user_unbought = {}
    for key in user_bought:
        user_unbought[key] = list(item_set - set(user_bought[key]))

    # 每个user的item按时间戳从后往前排序
    full_data['timestamp'] = full_data.groupby('user').rank(ascending=False)['timestamp']

    train_data = full_data[full_data['timestamp'] != 1].reset_index(drop=True)
    test_data = full_data[full_data['timestamp'] == 1].reset_index(drop=True)
    del train_data['timestamp']
    del test_data['timestamp']

    # 训练数据和标签, 由于是隐式反馈所有标签为1
    train_features = train_data
    train_labels = np.ones(len(train_data), dtype=np.int32).tolist()
    # 测试集返回的标签为item编号
    test_features = test_data
    test_labels = test_data['item'].tolist()

    return ((train_features, train_labels),
            (test_features, test_labels),
            (n_user, n_item),
            (user_bought, user_unbought))


def add_negative(features, user_unbought, labels, neg_num, is_training):
    feature_user, feature_item, labels_add, feature_dict = [], [], [], {}
    # 1个正样本对应neg_num个负样本
    for i in range(len(features)):
        user = features['user'][i]
        item = features['item'][i]
        label = labels[i]

        feature_user.append(user)
        feature_item.append(item)
        labels_add.append(label)
        # 负样本下采样, 这里有两个作用
        # 训练集：正负样本比例为1：n
        # 测试集：只要批量大小为1+n就可以产生1个已交互item和n个未交互item的数据
        neg_samples = np.random.choice(user_unbought[user], size=neg_num, replace=False).tolist()

        if is_training:
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                # 这里标签为0表示负样本
                labels_add.append(0)

        else:
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                # 这里标签为k表示itemID
                labels_add.append(k)

    feature_dict['user'] = feature_user
    feature_dict['item'] = feature_item

    return feature_dict, labels_add

# 保存数据, 字典格式
def dump_data(features, labels, user_unbought, num_neg, is_training):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    features, labels = add_negative(features, user_unbought, labels, num_neg, is_training)

    data_dict = dict([('user', features['user']),
                      ('item', features['item']), ('label', labels)])

    print(data_dict)
    if is_training:
        np.save(os.path.join(DATA_PATH, 'train_data.npy'), data_dict)
    else:
        np.save(os.path.join(DATA_PATH, 'test_data.npy'), data_dict)

# 训练集正负样本1：neg_num
def train_input_fn(features, labels, batch_size, user_unbought, num_neg):
    """ 构建训练集, 返回类型为tf.iterator"""
    data_path = os.path.join(DATA_PATH, 'train_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_unbought, num_neg, True)

    data = np.load(data_path).item()

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(100000).batch(batch_size)
    return dataset

# 测试集已交互item：未交互item = 1：n
def eval_input_fn(features, labels, user_unbought, test_neg):
    """ 构建测试集, 返回类型为tf.iterator"""
    data_path = os.path.join(DATA_PATH, 'test_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_unbought, test_neg, False)

    data = np.load(data_path).item()
    print("Loading testing data finished!")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # 每批量test_neg+1数据, 测试集中每个user只对应一个item对应这里的1
    # test_neg对应的是从这个user未交互的item随机抽取test_neg个item
    # 每次测试的时候都是拿某个user最后一次交互的item和test_neg个为交互的item去测试
    dataset = dataset.batch(test_neg + 1)

    return dataset
