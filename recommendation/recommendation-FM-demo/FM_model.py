from scipy.sparse import csr
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm

def vectorize_dic2(dic, ix=None, train_ref=None):
    if not train_ref:
        user_cnt = len(set(dic['users']))
        item_cnt = len(set(dic['items']))
    else:
        user_cnt = train_ref[0]
        item_cnt = train_ref[1]

    data_cnt = len(dic['users'])

    if ix is None:
        ix = {}
        # 如果没有提供索引字典就自己创建索引字典
        for id, num in enumerate(set(dic['users'])):
            ix['user' + str(num)] = id
        for id, num in enumerate(set(dic['items'])):
            ix['item' + str(num)] = user_cnt + id
    # 纵坐标
    col_ix = []
    for user_id, item_id in zip(dic['users'], dic['items']):
        # 如果user_id和item_id都在索引字典中
        if 'user' + str(user_id) in ix and 'item' + str(item_id) in ix:
            col_ix.extend([ix['user' + str(user_id)], ix['item' + str(item_id)]])
        # 至少一个不在索引字典中，则不使用这个数据
        else:
            data_cnt -= 1

    # 横坐标
    row_ix = np.repeat(np.arange(0, data_cnt), 2)
    data = np.ones(data_cnt * 2)

    print('col', len(col_ix))
    print('row', len(row_ix))
    print('data', len(data))
    return csr.csr_matrix((data, (row_ix, col_ix)), shape=(data_cnt, user_cnt + item_cnt)), ix, (user_cnt, item_cnt)


# 利用生成器产生批量数据
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:upper_bound]
        yield (ret_x, ret_y)


cols = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

# 利用users，items数据产生对应的稀疏矩阵
x_train, ix, train_ref = vectorize_dic2({'users': train['user'].values,
                                         'items': train['item'].values})
x_test, ix, _ = vectorize_dic2({'users': test['user'].values,
                                'items': test['item'].values}, ix, train_ref)
y_train = train['rating'].values
y_test = test['rating'].values

# 稀疏转稠密矩阵
x_train = x_train.todense()
x_test = x_test.todense()

print('The shape of x_train is: ', x_train.shape)
print('The shape of x_test is: ', x_test.shape)

# 训练数据shape
n, p = x_train.shape

# 占位符，存放x和y
x = tf.placeholder('float', [None, p])
y = tf.placeholder('float', [None, 1])

w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

# 每个特征对应一个k维向量
k = 10
v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))

# y_hat = tf.Variable(tf.zeros([n,1]))

# 线性项 w*x+w0
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))  # n * 1
# 交叉项
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(
            tf.matmul(x, tf.transpose(v)), 2),
        tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
    ), axis=1, keep_dims=True)

y_hat = tf.add(linear_terms, pair_interactions)

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

# L2正则化
l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w, tf.pow(w, 2)),
        tf.multiply(lambda_v, tf.pow(v, 2))
    )
)

# 损失+正则化
error = tf.reduce_mean(tf.square(y - y_hat))
loss = tf.add(error, l2_norm)

# 使用梯度下降法
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#
epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        # 打乱行号
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _, t = sess.run([train_op, loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print('train loss: %.4f' % t)

    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
    RMSE = np.sqrt(np.array(errors).mean())
    print('RMSE ERROR OF TEST DATA: %.4f' % RMSE)
