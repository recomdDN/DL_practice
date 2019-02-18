import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, args):

        self.is_training = args.is_training
        # 嵌入矩阵维度
        self.embedding_size = args.embedding_size
        # self.basic_size=args.basic_size
        # 低级品牌ID
        self.brand_list = args.brand_list
        # 中级品牌ID
        self.msort_list = args.msort_list
        # 商品总数量
        self.item_count = args.item_count
        # 低级品牌总数量
        self.brand_count = args.brand_count
        # 中级品牌总数量
        self.msort_count = args.msort_count
        self.build_model()

    def build_model(self):
        # 用户编号 [batch_size,]
        # self.u = tf.placeholder(tf.int32, [None, ])
        # 用户基本特征 [batch_size, basic_size]
        # self.basic = tf.placeholder(tf.float32, [None, 4])
        # 历史点击商品ID[batch_size, T]
        self.hist_click = tf.placeholder(tf.int32, [None, None])
        # 历史点击商品个数 [batch_size, ]
        self.sl = tf.placeholder(tf.int32, [None, ])
        # 最后一次点击商品ID[batch_size, 1]
        self.last_click = tf.placeholder(tf.int32, [None, 1])
        # 包含正样本和n个负样本, 用于softmax计算loss [batch_size, sub_sample_count]
        self.sub_sample = tf.placeholder(tf.int32, [None, None])
        # label one-hot[batch_size, class_num]
        self.y = tf.placeholder(tf.float32, [None, None])
        # drop_out 概率
        self.keep_prob = tf.placeholder(tf.float32, [])
        # 学习率
        self.lr = tf.placeholder(tf.float64, [])

        # emb variable wx+b
        # 商品id嵌入矩阵
        item_emb_w = tf.get_variable("item_emb_w", [self.item_count, self.embedding_size])
        # 低级品牌类别id嵌入矩阵
        brand_emb_w = tf.get_variable("brand_emb_w", [self.brand_count, self.embedding_size])
        # 中级品牌类别id嵌入矩阵
        msort_emb_w = tf.get_variable("msort_emb_w", [self.msort_count, self.embedding_size])

        input_b = tf.get_variable("input_b", [self.item_count], initializer=tf.constant_initializer(0.0))

        brand_list = tf.convert_to_tensor(self.brand_list, dtype=tf.int32)
        msort_list = tf.convert_to_tensor(self.msort_list, dtype=tf.int32)

        # 根据历史点击商品ID获取对应商品的低级品牌ID和中级品牌ID
        hist_brand = tf.gather(brand_list, self.hist_click)
        hist_msort = tf.gather(msort_list, self.hist_click)
        # 嵌入向量拼接[batch_size, max_sl, 3*embedding_size]
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_click),
                           tf.nn.embedding_lookup(brand_emb_w, hist_brand),
                           tf.nn.embedding_lookup(msort_emb_w, hist_msort)], axis=2)

        # 用mask去除无用的点击历史 [batch_size, max_sl]
        mask = tf.sequence_mask(lengths=self.sl, maxlen=tf.shape(h_emb)[1], dtype=tf.float32)
        # [batch_size, max_sl, 1]
        mask = tf.expand_dims(mask, -1)
        # [batch_size, max_sl, 3 * embedding_size]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])
        # [batch_size, max_sl, 3 * embedding_size]
        h_emb *= mask
        # 这里没有用tf.reduce_mean是因为这个函数除以固定长度max_sl而不是对应历史商品个数, \
        # 这会导致历史点击商品个数较少的用户得到的历史嵌入向量数值过小
        # 历史值求和[batch_size, 3 * embedding_size]
        hist = tf.reduce_sum(h_emb, 1)
        # 除以相应历史点击商品个数[batch_size, 3 * embedding_size]
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, 3 * self.embedding_size]), tf.float32))

        # 最后一次点击的商品ID [batch_size, 1]
        last_brand = tf.gather(brand_list, self.last_click)
        last_msort = tf.gather(msort_list, self.last_click)
        last_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.last_click),
                              tf.nn.embedding_lookup(brand_emb_w, last_brand),
                              tf.nn.embedding_lookup(msort_emb_w, last_msort)], axis=-1)
        # [batch_size, 3*embedding_size]
        last_emb = tf.squeeze(last_emb, axis=1)

        # 将历史点击商品均值与最后一次点击的商品拼接
        # self.input = tf.concat([hist, last_emb, self.basic], axis=-1)
        self.input = tf.concat([hist, last_emb], axis=-1)

        # 3层全连接网络
        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)
        # 当做用户向量
        layer_3 = tf.layers.dense(layer_2, 3 * self.embedding_size, activation=tf.nn.relu, name='f3')

        # 训练 直接用输入商品嵌入向量与用户向量做内积再通过softmax来获取输出概率
        # 这里并没有使用输入嵌入向量作为商品向量, 主要是因为输入嵌入向量解释性更好
        if self.is_training:

            # 根据下采样负样本商品ID获取对应商品的低级品牌ID和中级品牌ID
            sam_brand = tf.gather(brand_list, self.sub_sample)
            sam_msort = tf.gather(msort_list, self.sub_sample)

            # [batch_size, sub_sample_count]
            sample_b = tf.nn.embedding_lookup(input_b, self.sub_sample)
            # 获取下采样样本的输入嵌入向量[batch_size, sub_sample_count, 3 * embedding_size]
            sample_w = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.sub_sample),
                                  tf.nn.embedding_lookup(brand_emb_w, sam_brand),
                                  tf.nn.embedding_lookup(msort_emb_w, sam_msort)
                                  # tf.tile(tf.expand_dims(self.basic, 1), [1, tf.shape(sample_b)[1], 1])
                                  ], axis=2)
            # [batch_size, 1, 3*embedding_size]
            user_v = tf.expand_dims(layer_3, 1)
            # [batch_size, 3*embedding_size, sub_sample_num]
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])

            self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b

            # Step variable
            # global_step记录训练批次(batch), global_epoch_step记录训练轮次(epoch)
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            self.yhat = tf.nn.softmax(self.logits)

            self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))

            trainable_params = tf.trainable_variables()
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

        # 预测所有商品
        else:
            # 所有商品的输入嵌入 [item_count, 3*embedding_size]
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(brand_emb_w, brand_list),
                                 tf.nn.embedding_lookup(msort_emb_w, msort_list)],
                                axis=1)
            # 用户与商品做内积 [item_count, 1]
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True) + input_b
            self.output = tf.nn.softmax(self.logits)

    def train(self, sess, uij, l, keep_prob):
        loss, yhat, _ = sess.run([self.loss, self.yhat, self.train_op], feed_dict={
            # self.u: uij[0],
            self.sub_sample: uij[1],
            self.y: uij[2],
            self.hist_click: uij[3],
            self.sl: uij[4],
            # self.basic: uij[5],
            self.last_click: uij[6],
            self.lr: l,
            self.keep_prob: keep_prob
        })
        gauc, r = Model.cal_gauc(yhat)
        return loss, gauc, r

    def eval(self, sess, uij, keep_prob):
        loss, yhat = sess.run([self.loss, self.yhat], feed_dict={
            # self.u: uij[0],
            self.sub_sample: uij[1],
            self.y: uij[2],
            self.hist_click: uij[3],
            self.sl: uij[4],
            # self.basic: uij[5],
            self.last_click: uij[6],
            self.keep_prob: keep_prob
        })
        gauc, r = Model.cal_gauc(yhat)
        return loss, gauc, r

    @ staticmethod
    def cal_gauc(y_pred):
        r, c = np.shape(y_pred)
        y_pos = y_pred[:, 0]
        y_neg = y_pred[np.arange(r), np.random.randint(1, c, r)]
        gauc = np.mean((y_pos - y_neg) > 0)
        return gauc, r

    def test(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            # self.u: uij[0],
            self.hist_click: uij[1],
            self.sl: uij[2],
            # self.basic: uij[3],
            self.last_click: uij[4],
            self.keep_prob: keep_prob
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
