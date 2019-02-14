import random
import pickle
import numpy as np

random.seed(1234)

with open('../raw_data/reviews.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    # 用户ID, 商品ID, 评论时间
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
with open('../raw_data/meta.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    # 商品ID, 商品所属类别
    meta_df = meta_df[['asin', 'categories']]
    # 只选最后一个类别
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

# 将商品ID, 商品类别, 用户ID重新从0开始编号
asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

# 商品ID升排序
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
# 按用户和时间升排序
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

with open('../raw_data/remap.pkl', 'wb') as f:
    # 用户id, 商品id, 评论时间
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    # 商品类别list
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
    # 用户数, 商品数, 类别数, 样本数
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    # 原编号
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
