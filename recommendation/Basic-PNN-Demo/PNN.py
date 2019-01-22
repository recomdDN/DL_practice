import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class PNN(BaseEstimator, TransformerMixin):

    def __init__(self, n_feature, n_field,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size=50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 greater_is_better=True,
                 use_inner=True):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        # 特征数量
        self.n_feature = n_feature
        # field数量
        self.n_field = n_field
        # 嵌入维度
        self.embedding_size = embedding_size
        # DNN每层神经元个数
        self.deep_layers = deep_layers
        # 论文中lz和lp的维度
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        # 激活函数
        self.deep_layers_activation = deep_layer_activation
        # 训练轮数
        self.epoch = epoch
        # 批量大小
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        # 优化方法
        self.optimizer_type = optimizer
        # batch_normalization
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        # 损失类型
        self.loss_type = loss_type
        # 评价函数
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self.use_inner = use_inner

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            # 特征索引
            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None, None],
                                             name='feat_index')
            # 特征值
            self.feat_value = tf.placeholder(tf.float32,
                                             shape=[None, None],
                                             name='feat_value')
            # 标签值
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            # dropout概率
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            # 初始化参数
            self.weights = self._initialize_weights()

            # Embeddings N * n_field * embedding_size
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.n_field, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # 线性项
            linear_output = []
            for i in range(self.deep_init_size):
                linear_output.append(tf.reshape(
                    tf.reduce_sum(tf.multiply(self.embeddings, self.weights['product-linear'][i]), axis=[1, 2]),
                    shape=(-1, 1)))  # N * 1

            self.lz = tf.concat(linear_output, axis=1)  # N * init_deep_size

            # 二次项
            quadratic_output = []
            # 基于内积
            if self.use_inner:
                for i in range(self.deep_init_size):
                    quadratic_output.append(tf.reshape(tf.norm(
                        tf.tensordot(self.embeddings, self.weights['product-quadratic-inner'][i], (1, 1)),
                        [-1, self.embedding_size], axis=(1, 2)), shape=(-1, 1)))

            # 基于外积
            else:
                embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
                p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))  # N * K * K
                for i in range(self.deep_init_size):
                    theta = tf.multiply(p, tf.expand_dims(self.weights['product-quadratic-outer'][i], 0))  # N * K * K
                    quadratic_output.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))  # N * 1

            self.lp = tf.concat(quadratic_output, axis=1)  # N * init_deep_size

            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            # DNN
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            self.out = tf.add(tf.matmul(self.y_deep, self.weights['output']), self.weights['output_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
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
            tf.random_normal([self.n_feature, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')
        # Product Layers
        # 线性项
        weights['product-linear'] = tf.Variable(
            tf.random_normal([self.deep_init_size, self.n_field, self.embedding_size], 0.0, 0.01))
        # 二次项
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.n_field], 0.0, 0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.embedding_size, self.embedding_size], 0.0, 0.01))
        # 线性项和二次项求和的偏置项
        weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size, ], 0, 0, 1.0))
        # DNN
        num_layer = len(self.deep_layers)
        # DNN层
        for i in range(num_layer):
            last_layers = self.deep_layers[i - 1] if i > 0 else self.deep_init_size
            glorot = np.sqrt(2.0 / (last_layers + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(last_layers, self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
        weights['output'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                        dtype=np.float32)
        weights['output_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def predict(self, Xi, Xv, y):
        """
        :param Xi: 特征索引
        :param Xv: 特征值
        :return: 损失值
        """
        # dummy y
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_dep,
                     self.train_phase: True}

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None):
        """
        训练遍历数据集epoch次
        :param Xi_train: 训练集索引
        :param Xv_train: 训练集特征值
        :param y_train: 训练集标签值
        :param Xi_valid: 验证集索引
        :param Xv_valid: 验证集特征值
        :param y_valid: 验证集标签值
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                val_loss = self.predict(Xi_valid, Xv_valid, y_valid)
                print("epoch", epoch, "val-loss: %.4f" % val_loss[0])
