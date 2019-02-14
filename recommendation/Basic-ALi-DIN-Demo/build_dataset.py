import random
import pickle

random.seed(1234)

with open('raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    # 某个用户对应的浏览过的商品
    pos_list = hist['asin'].tolist()

    # 返回一个未浏览过的商品
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    # 已浏览商品与未浏览商品1:1
    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i != len(pos_list) - 1:
            # 数据格式(用户ID, [前i个历史浏览商品列表], 第i个已浏览商品, 1)
            # [u, hist_i, i, y]
            train_set.append((reviewerID, hist, pos_list[i], 1))
            # 数据格式(用户ID, [前i个历史浏览商品列表], 第i个未浏览商品, 0)
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            # 把最后一组数据当作测试数据
            test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(test_set)

# 判断测试集中是不是每个用户都有一条数据
assert len(test_set) == user_count


with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
