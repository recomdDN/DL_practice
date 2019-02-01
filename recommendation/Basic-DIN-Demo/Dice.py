import tensorflow as tf


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    """
    dice激活层, 主要是计算x_p,
    1、对输入进行标准化，2、进行sigmoid变换
    :param _x: 输入层 shape = [..., last_shape]
    :param axis: 这个轴表示一个数据的所有特征
    :param epsilon:
    :param name: 变量后缀
    :return: 返回与输入shape一样的张量
    """
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta' + name, _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)
    # batchNorm实现
    # 输入的shape
    # input_shape = list(_x.get_shape())
    # # 需要缩减的维度, 缩减x的前n-1维
    # reduction_axes = list(range(len(input_shape)))
    # del reduction_axes[axis]
    #
    # # 需要重塑的shape, [1,...,1,last_shape]
    # broadcast_shape = [1] * len(input_shape)
    # broadcast_shape[axis] = input_shape[axis]
    #
    # # [last_shape,]
    # mean = tf.reduce_mean(_x, axis=reduction_axes)
    # # [1,...,1,last_shape]
    # brodcast_mean = tf.reshape(mean, broadcast_shape)
    #
    # std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    # std = tf.sqrt(std)
    # brodcast_std = tf.reshape(std, broadcast_shape)
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # tensorflow自带batchNorm
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def parametric_relu(_x):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
