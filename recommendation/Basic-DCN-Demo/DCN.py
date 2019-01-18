import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

#
class DCN(BaseEstimator, TransformerMixin):

    def __init__(self, n_cate_feature, n_field, n_numeric_feature,
                 embedding_size=8,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True, cross_layer_num=3):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        # 类别型特征个数
        self.n_cate_feature = n_cate_feature
        # 数值型特征数量
        self.n_numeric_feature = n_numeric_feature
        # field数量
        self.n_field = n_field
        # 嵌入维度
        self.embedding_size = embedding_size
        # 总特征数
        self.total_size = self.n_field * self.embedding_size + self.n_numeric_feature
        # DNN各层隐藏神经元数
        self.deep_layers = deep_layers
        # cross层数
        self.cross_layer_num = cross_layer_num
        # DNN每层dropout概率
        self.dropout_dep = dropout_deep
        # DNN激活函数
        self.deep_layers_activation = deep_layers_activation
        # L2正则化参数
        self.l2_reg = l2_reg
        # 训练轮数
        self.epoch = epoch
        # 批量大小
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        # 优化方式
        self.optimizer_type = optimizer_type
        # batch_norm大小
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        # 损失类型
        self.loss_type = loss_type
        # 评价准则
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # [None, n_field]
            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None, None],
                                             name='feat_index')
            # [None, n_field]
            self.feat_value = tf.placeholder(tf.float32,
                                             shape=[None, None],
                                             name='feat_value')
            # 数值特征[None, n_numeric_feature]
            self.numeric_value = tf.placeholder(tf.float32, [None, None], name='num_value')
            # 标签值
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            # DNN, dropout概率
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            # 初始化权重
            self.weights = self._initialize_weights()

            # 获取稀疏特征对应的嵌入向量
            # batch_size * field * embedding_size
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.n_field, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
            # 将稠密向量和稀疏向量嵌入后的向量拼接
            self.x0 = tf.concat([self.numeric_value,
                                 tf.reshape(self.embeddings, shape=[-1, self.n_field * self.embedding_size])]
                                , axis=1)

            # DNN部分
            self.x_deep = tf.nn.dropout(self.x0, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.x_deep = tf.add(tf.matmul(self.x_deep, self.weights["deep_layer_%d" % i]),
                                     self.weights["deep_bias_%d" % i])
                self.x_deep = self.deep_layers_activation(self.x_deep)
                self.x_deep = tf.nn.dropout(self.x_deep, self.dropout_keep_deep[i + 1])

            # 交叉网络部分
            self._x0 = tf.reshape(self.x0, (-1, self.total_size, 1))
            x_l = self._x0
            for l in range(self.cross_layer_num):
                x_l = tf.tensordot(tf.matmul(self._x0, x_l, transpose_b=True),
                                   self.weights["cross_layer_%d" % l], 1) + self.weights["cross_bias_%d" % l] + x_l

            self.cross_network_out = tf.reshape(x_l, (-1, self.total_size))

            # 拼接全神经网络和交叉网络的输出
            concat_input = tf.concat([self.cross_network_out, self.x_deep], axis=1)

            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 正则化
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["deep_layer_%d" % i])
                for i in range(self.cross_layer_num):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["cross_layer_%d" % i])

            # 优化方法
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.n_cate_feature, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.n_cate_feature, 1], 0.0, 1.0),
                                              name='feature_bias')

        # DNN权重
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0 / (self.total_size + self.deep_layers[0]))

        weights['deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.total_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # 交叉网络权重
        for i in range(self.cross_layer_num):
            weights["cross_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size, 1)),
                dtype=np.float32)
            weights["cross_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size, 1)),
                dtype=np.float32)  # 1 * layer[i]

        # 最后一层的神经元个数
        input_size = self.total_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                                   dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    # 获取批数据
    def get_batch(self, Xi, Xv, Xv2, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xv2[start:end], [[y_] for y_ in y[start:end]]

    # 同时打乱4个list
    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    # 计算新样本的损失
    def predict(self, Xi, Xv, Xv2, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y

        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    # 批量训练
    def fit_on_batch(self, Xi, Xv, Xv2, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_dep,
                     self.train_phase: True}
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train,
            cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv_valid=None, y_valid=None):
        """ fit只完成一次训练数据上的训练
        :param cate_Xi_train: 训练集类别型特征索引
        :param cate_Xv_train: 训练集类别型特征
        :param numeric_Xv_train: 训练集数值型特征
        :param y_train: 训练集标签值
        :param cate_Xi_valid:　验证集类别型特征索引
        :param cate_Xv_valid: 验证集类别型特征
        :param numeric_Xv_valid: 验证集数值型特征
        :param y_valid: 验证集标签
        """
        print(len(cate_Xi_train))
        print(len(cate_Xv_train))
        print(len(numeric_Xv_train))
        print(len(y_train))
        has_valid = cate_Xv_valid is not None
        for epoch in range(self.epoch):
            self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train)
            # 全部数据可以分为多少个数据批
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch, numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train,
                                                                                         numeric_Xv_train, y_train,
                                                                                         self.batch_size, i)

                self.fit_on_batch(cate_Xi_batch, cate_Xv_batch, numeric_Xv_batch, y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                loss = self.predict(cate_Xi_valid, cate_Xv_valid, numeric_Xv_valid, y_valid)
                print("epoch", epoch, "val-loss", loss)
