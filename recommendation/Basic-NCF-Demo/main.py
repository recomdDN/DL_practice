import os, sys, time
import numpy as np
import tensorflow as tf

import NCF_input
import NCF
import metrics

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, 'size of mini-batch.')
tf.app.flags.DEFINE_integer('negative_num', 4, 'number of negative samples.')
tf.app.flags.DEFINE_integer('test_neg', 99, 'number of negative samples for test.')
tf.app.flags.DEFINE_integer('embedding_size', 16, 'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('epochs', 20, 'the number of epochs.')
tf.app.flags.DEFINE_integer('topK', 10, 'topk for evaluation.')
tf.app.flags.DEFINE_string('optim', 'Adam', 'the optimization method.')
tf.app.flags.DEFINE_string('initializer', 'Xavier', 'the initializer method.')
tf.app.flags.DEFINE_string('loss_func', 'cross_entropy', 'the loss function.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')

tf.app.flags.DEFINE_string('model_dir', 'model/', 'the dir for saving model.')
tf.app.flags.DEFINE_float('regularizer', 0.0, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')


def train(train_data, test_data, n_user, n_item):
    with tf.Session() as sess:
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        model = NCF.NCF(FLAGS.embedding_size, n_user, n_item, FLAGS.lr,
                        FLAGS.optim, FLAGS.initializer, FLAGS.loss_func, FLAGS.activation,
                        FLAGS.regularizer, iterator, FLAGS.topK, FLAGS.dropout, is_training=True)

        model.build()

        # 有参数就读取, 没有就重新训练
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            # 加载模型参数
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        count = 0
        for epoch in range(FLAGS.epochs):
            # 训练集的迭代器
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            model.get_data()
            start_time = time.time()

            try:
                while True:  # 直到生成器没数据, 也就是所有训练数据遍历一次
                    model.step(sess, count)
                    count += 1
            except tf.errors.OutOfRangeError:
                # 打印训练一轮的时间
                print("Epoch %d training " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))
            # 测试集的迭代器
            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            model.get_data()
            start_time = time.time()
            HR, MRR, NDCG = [], [], []
            pred_item, gt_item = model.step(sess, None)
            try:
                while True:  # 直到生成器没数据, 也就是所有测试数据遍历一次
                    pred_item, gt_item = model.step(sess, None)

                    gt_item = int(gt_item[0])
                    HR.append(metrics.hit(gt_item, pred_item))
                    MRR.append(metrics.mrr(gt_item, pred_item))
                    NDCG.append(metrics.ndcg(gt_item, pred_item))
            # 评估值取均值
            except tf.errors.OutOfRangeError:
                hr = np.array(HR).mean()
                mrr = np.array(MRR).mean()
                ndcg = np.array(NDCG).mean()
                print("Epoch %d testing  " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))
                print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

        # 保存模型参数
        checkpoint_path = os.path.join(FLAGS.model_dir, "NCF.ckpt")
        model.saver.save(sess, checkpoint_path)


def main():
    ((train_features, train_labels),
     (test_features, test_labels),
     (n_user, n_item),
     (user_bought, user_unbought)) = NCF_input.load_data()

    print(train_features[:10])
    # 训练数据
    train_data = NCF_input.train_input_fn(train_features, train_labels, FLAGS.batch_size, user_unbought,
                                          FLAGS.negative_num)
    # 测试数据
    test_data = NCF_input.eval_input_fn(test_features, test_labels,
                                        user_unbought, FLAGS.test_neg)
    # 训练模型
    train(train_data, test_data, n_user, n_item)


if __name__ == '__main__':
    main()
