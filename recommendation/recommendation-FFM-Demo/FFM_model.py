import tensorflow as tf
import numpy as np
import os

# 特征数量
feature_cnt = 20
# field个数
field_cnt = 2
# 隐向量长度
vector_dimension = 3

total_plan_train_steps = 1000
# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 100

# 样本量
data_cnt = 1000

lr = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"

# 定义参数
def create_latent_weight(feature_cnt, field_cnt, vector_dimension):
    weights = tf.truncated_normal([feature_cnt, field_cnt, vector_dimension])
    tf_weights = tf.Variable(weights)
    return tf_weights

def create_one_dimension_weight(feature_cnt):
    weights = tf.truncated_normal([feature_cnt, 1])
    tf_weights = tf.Variable(weights)
    return tf_weights

def create_bias():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)
    return tf_weights


def inference(input_x, input_x_field, zeroWeights, oneDimWeights, thirdWeight):
    """计算回归模型输出的值"""
    # w*x
    secondValue = tf.matmul(input_x, oneDimWeights, name='secondValue')
    # w*x+b 一次项
    firstTwoValue = tf.add(zeroWeights, secondValue, name="firstTwoValue")
    # 二次项
    thirdValue = []

    print('third')
    for i in range(feature_cnt-1):
        # 第一个交叉特征对应的索引i
        featureIndex1 = i
        # 第i个特征对应的field
        fieldIndex1 = int(input_x_field[i])
        for j in range(i + 1, feature_cnt):
            # 第二个交叉特征对应的索引j
            featureIndex2 = j
            # 第j个特征对应的field
            fieldIndex2 = int(input_x_field[j])
            # # 提取出特征1对应的field
            # 隐向量1对应的索引[特征1索引，特征2所属field，隐向量维度]
            vectorLeft = tf.convert_to_tensor([[featureIndex1, fieldIndex2, i] for i in range(vector_dimension)])
            # 分片索引，shape=[vector_dimension,]
            weightLeft = tf.gather_nd(thirdWeight, vectorLeft)

            # # 提取出特征2对应的field
            # 隐向量2对应的索引[特征2索引，特征1所属field，隐向量维度]
            vectorRight = tf.convert_to_tensor([[featureIndex2, fieldIndex1, i] for i in range(vector_dimension)])
            # 分片索引，[vector_dimension,]
            weightRight = tf.gather_nd(thirdWeight, vectorRight)
            # transpose(vi,fj) * vj,fi
            tempValue = tf.reduce_sum(tf.multiply(weightLeft, weightRight))

            feature1_index = tf.convert_to_tensor([[row, i] for row in range(batch_size)])
            feature2_index = tf.convert_to_tensor([[row, j] for row in range(batch_size)])

            xi = tf.gather_nd(input_x, feature1_index)
            xj = tf.gather_nd(input_x, feature2_index)
            product = tf.reshape(tf.multiply(xi, xj), [batch_size, -1])

            secondItemVal = tf.multiply(tempValue, product)

            thirdValue.append(secondItemVal)
    return tf.add(firstTwoValue, tf.reduce_sum(thirdValue, 0))


def gen_data():
    labels = [-1, 1]
    # 随机从-1,1中选一个作为标签
    y = [np.random.choice(labels, 1)[0] for _ in range(data_cnt)]
    # 前一半field为0，后一半field为1
    x_field = [i // 10 for i in range(feature_cnt)]
    # 随机初始化特征值
    x = np.random.randint(0, 2, size=(data_cnt, feature_cnt))
    return x, y, x_field


if __name__ == '__main__':
    global_step = tf.Variable(0, trainable=False)
    trainx, trainy, trainx_field = gen_data()
    #
    input_x = tf.placeholder(tf.float32, [None, feature_cnt])
    input_y = tf.placeholder(tf.float32, [None, ])


    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    zeroWeights = create_bias()
    oneDimWeights = create_one_dimension_weight(feature_cnt)
    # 创建二次项的权重变量 n * f * k
    thirdWeight = create_latent_weight(feature_cnt,
                                           field_cnt,
                                           vector_dimension)

    y_ = inference(input_x, trainx_field, zeroWeights, oneDimWeights, thirdWeight)
    # L2正则化
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(oneDimWeights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(thirdWeight, 2)), axis=[1, 2])
        )
    )

    loss = tf.reduce_mean(tf.log(1 + tf.exp(-input_y * y_))) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(total_plan_train_steps):
            for t in range(0, data_cnt, batch_size):
                end = t+batch_size if t+batch_size < data_cnt else data_cnt
                input_x_batch = trainx[t:end]
                input_y_batch = trainy[t:end]
                predict_loss, _ = sess.run([loss, train_step],
                                           feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After  {step} training step(s), loss on training batch is {predict_loss} "
                      .format(step=i, predict_loss=predict_loss))

                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()
        # 保存模型
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
