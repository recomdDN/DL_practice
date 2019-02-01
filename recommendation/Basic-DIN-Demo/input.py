import numpy as np

# 数据迭代器
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

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            # 用户ID
            u.append(t[0])
            # 商品ID
            i.append(t[2])
            # 标签
            y.append(t[3])
            # sl代表历史浏览商品个数
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        # 统一各个样本已浏览商品列表长度
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        # 轮次, (用户ID, 商品ID, 标签, 历史商品ID列表, 历史商品个数)
        return self.i, (u, i, y, hist_i, sl)


class DataInputTest:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        # 记录当前是第几batch数据
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size:min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, i, j, sl = [], [], [], []

        for t in ts:
            # 用户ID
            u.append(t[0])
            # 已浏览商品ID
            i.append(t[2][0])
            # 未浏览商品ID
            j.append(t[2][1])
            # 已浏览的商品个数
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        # 统一各个样本已浏览商品列表长度
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        # 轮次, (用户ID, 已浏览商品ID, 未浏览商品ID, 历史商品ID列表, 历史商品个数)
        return self.i, (u, i, j, hist_i, sl)
