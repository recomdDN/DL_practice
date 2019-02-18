import numpy as np


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        # 总轮次
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, sub_sample, y, sl, basic, last_click, = [], [], [], [], [], []
        for t in ts:
            # 用户ID [batch_size,]
            u.append(t[0])
            # 下采样样本 = 最后一次点击商品 + 20个未点击商品 [batch_size, 21]
            sub_sample.append([t[2]] + t[3])
            # 下采样样本个数 21
            sub_sample_size = len(t[3]) + 1
            mask = np.zeros(sub_sample_size, np.int64)
            mask[0] = 1
            # 标签值 [batch_size, sub_sample_size]
            y.append(mask)
            # 历史点击商品个数 [batch_size,]
            sl.append(len(t[1]))
            # 用户基本特征
            # basic.append(t[4])
            # last_click [batch_size, 1]
            last_click.append([t[2]])
        # 当前数据批最大历史点击商品个数
        max_sl = max(sl)
        # [batch_size, max_sl]
        hist_click = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_click[k][l] = t[1][l]
            k += 1
        return self.i, (u, sub_sample, y, hist_click, sl, basic, last_click)

class DataInputPredict:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, sl, basic, last_click = [], [], [], []
        for t in ts:
            # 用户ID
            u.append(t[0])
            # 历史点击商品ID
            sl.append(len(t[1]))
            # 用户基本特征
            # basic.append(t[2])
            # last_click
            last_click.append([t[3]])

        max_sl = max(sl)

        hist_click = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_click[k][l] = t[1][l]
            k += 1

        return self.i, (u, hist_click, sl, basic, last_click)
