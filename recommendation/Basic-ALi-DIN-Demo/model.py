import tensorflow as tf
from Dice import dice


class Model(object):

    def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
        # 用户ID
        self.u = tf.placeholder(tf.int32, [None, ])  # [batch_size, ]
        # 已浏览商品ID
        self.i = tf.placeholder(tf.int32, [None, ])  # [batch_size, ]
        # 未浏览商品ID
        self.j = tf.placeholder(tf.int32, [None, ])  # [batch_size, ]
        # 标签
        self.y = tf.placeholder(tf.float32, [None, ])  # [batch_size, ]
        # 历史浏览商品
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [batch_size, T]
        # 历史浏览商品个数
        self.sl = tf.placeholder(tf.int32, [None, ])  # [batch_size, ]
        # 学习率
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128
        # 用户嵌入矩阵
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        # 商品嵌入矩阵
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        # 商品类别嵌入矩阵
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])

        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        # 商品类别[batch_size, ]
        ic = tf.gather(cate_list, self.i)
        # 商品嵌入向量[batch_size, hidden_units]
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
        i_b = tf.gather(item_b, self.i)

        # 商品类别[batch_size, ]
        jc = tf.gather(cate_list, self.j)
        # 商品嵌入向量[batch_size, hidden_units]
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        j_b = tf.gather(item_b, self.j)

        # 历史商品类别[batch_size, max_sl]
        hc = tf.gather(cate_list, self.hist_i)
        # 历史商品嵌入向量[batch_size, max_sl, hidden_units]
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

        # attention层 [batch_size, 1, hidden_size]
        hist_i = attention(i_emb, h_emb, self.sl)
        # BN和全连接层
        hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')
        # [batch_size, hidden_units]
        h_emb_i = hist_i

        # attention层 [batch_size, 1, hidden_size]
        hist_j = attention(j_emb, h_emb, self.sl)
        # BN和全连接层
        hist_j = tf.layers.batch_normalization(inputs=hist_j)
        hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)
        # [batch_size, hidden_units]
        h_emb_j = hist_j

        # 3层全连接层(80, 40, 1) 输出shape = [batch_size, 1]
        din_i = tf.concat([h_emb_i, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.relu, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.relu, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        # [batch_size, 2*hidden_units]
        din_j = tf.concat([h_emb_j, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        # 预测用(80, 40, 1) 输出shape = [batch_size, 1]
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.relu, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.relu, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        # [batch_size,]
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        # logits_i - logits_j, size=[batch_size,]
        x = i_b + d_layer_3_i - d_layer_3_j - j_b
        # wx + b [batch_size, 1]
        self.logits = i_b + d_layer_3_i
        # 预测备选商品
        # 所有商品的ID嵌入向量和类别嵌入向量 [item_counts, hidden_units]
        item_emb_all = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        # 只取前predict_ads_num个备选商品[predict_ads_num, hidden_units]
        item_emb_sub = item_emb_all[:predict_ads_num, :]
        item_emb_sub = tf.expand_dims(item_emb_sub, 0)
        # [predict_batch_size, predict_ads_num, hidden_units]
        item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])
        # 将每个样本的历史商品列表与predict_ads_num个item求attention
        # [predict_batch_size, predict_ads_num, hidden_units]
        hist_sub = attention_multi_items(item_emb_sub, h_emb, self.sl)
        # -- attention end ---

        # BN和全连接层
        hist_sub = tf.layers.batch_normalization(inputs=hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
        # [predict_batch_size * predict_ads_num, hidden_units]
        hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
        hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

        # [predict_batch_size * predict_ads_num, hidden_units]
        u_emb_sub = hist_sub
        # [predict_batch_size * predict_ads_num, hidden_units]
        item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
        # [predict_batch_size * predict_ads_num, 2*hidden_units]
        din_sub = tf.concat([u_emb_sub, item_emb_sub], axis=-1)
        din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
        d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.relu, name='f1', reuse=True)
        d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.relu, name='f2', reuse=True)
        d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
        # [predict_batch_size, predict_ads_num]
        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])

        self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
        # [predict_batch_size, predict_ads_num, 1]
        self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
        # -- fcn end -------

        # 评价指标1 用户AUC
        # 简单来说其实就是基于同一用户随机抽出一对样本（一个正样本，一个负样本）
        # ，然后用训练得到的分类器来对这两个样本进行预测，预测得到正样本的概率大于负样本概率的概率
        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        # 概率值
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])

        # 评价指标2 [predict_batch_size, 2]
        # 用来计算整个数据集的AUC，抽取一对样本的时候没有基于同一用户
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

        # 当前训练步数
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # epoch轮数
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )
        # 可训练参数
        trainable_params = tf.trainable_variables()
        # 梯度截断
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        # 这里会自动完成global_step自动加1操作
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
        })
        return loss

    def eval(self, sess, uij):
        u_auc, score_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, score_p_and_n

    def test(self, sess, uij):
        return sess.run(self.logits_sub, feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    '''
      :param queries: [batch_size, hidden_units]
      :param keys: [batch_size, max_sl, hidden_units]
      :param keys_length: [batch_size]
      :return: [batch_size, 1, hidden_units]
    '''
    # queries隐层神经元个数
    queries_hidden_units = queries.get_shape().as_list()[-1]

    # 将queries拷贝成[batch_size, max_sl * hidden_units]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    # [batch_size, max_sl, hidden_units]
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    # 这里比原论文增加了乘积项 [batch_size, max_sl, 4 * hidden_units]

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # [batch_size, max_sl, 80]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    # [batch_size, max_sl, 40]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    # [batch_size, max_sl, 1]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    # [batch_size, 1, max_sl]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask # [batch_size, max_sl]
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    # [batch_size, 1, max_sl]
    key_masks = tf.expand_dims(key_masks, 1)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    # 元素替换 [batch_size, 1, max_sl]
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation # [batch_size, 1, max_sl]
    outputs = tf.nn.softmax(outputs)

    # Weighted sum # [batch_size, 1, hidden_units]
    outputs = tf.matmul(outputs, keys)

    return outputs


def attention_multi_items(queries, keys, keys_length):
    '''
      :param queries: [batch_size, ads_num, hidden_units] ads：广告数量
      :param keys: [batch_size, max_sl, hidden_units]
      :param keys_length: [batch_size]
      :return: [batch_size, ads_num, hidden_units]
    '''
    # queries隐层神经元个数
    queries_hidden_units = queries.get_shape().as_list()[-1]

    # ads个数
    queries_nums = queries.get_shape().as_list()[1]
    # 将queries拷贝成[batch_size, ads_num, max_sl * hidden_units]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    # [batch_size, ads_num, max_sl, hidden_units]
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]

    max_len = tf.shape(keys)[1]
    # 将keys拷贝成[batch_size, ads_num * max_sl, hidden_units]
    keys = tf.tile(keys, [1, queries_nums, 1])
    # [batch_size, ads_num, max_sl, hidden_units]
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    # [batch_size, ads_num, max_sl, 4 * hidden_units]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # [batch_size, ads_num, max_sl, 80]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    # [batch_size, ads_num, max_sl, 40]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    # # [batch_size, ads_num, max_sl, 1]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    # [batch_size, ads_num, 1, max_sl]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all
    # Mask # [batch_size, max_sl]
    key_masks = tf.sequence_mask(keys_length, max_len)
    # [batch_size * ads_num, max_sl]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    # [batch_size, ads_num, 1, max_sl]
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    # 元素替换 [batch_size, ads_num, 1, max_sl]
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    # [batch_size, ads_num, 1, max_sl]
    outputs = tf.nn.softmax(outputs)
    # [batch_size * ads_num, 1, max_sl]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    # [batch_size * ads_num, max_sl, hidden_units]
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])

    # Weighted sum # [batch_size * ads_num, 1, hidden_units]
    outputs = tf.matmul(outputs, keys)
    # [batch_size, ads_num, hidden_units]
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])

    return outputs
