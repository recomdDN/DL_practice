import os
import time
import pickle
import sys
import random
from input import DataInput, DataInputPredict
import numpy as np
import tensorflow as tf
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Args:
    is_training = True
    is_eval = True
    is_predict = False
    embedding_size = 256
    brand_list = None
    msort_list = None
    item_count = -1
    brand_count = -1
    msort_count = -1


if __name__ == '__main__':
    # read data
    with open('../raw_data/dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        brand_list = pickle.load(f)
        msort_list = pickle.load(f)
        user_count, item_count, brand_count, msort_count = pickle.load(f)
        item_key, brand_key, msort_key, user_key = pickle.load(f)
    print('user_count: %d\titem_count: %d\tbrand_count: %d\tmsort_count: %d' %
          (user_count, item_count, brand_count, msort_count))

    print('train set size', len(train_set))

    # init args
    args = Args()
    args.brand_list = brand_list
    args.msort_list = msort_list
    args.item_count = item_count
    args.brand_count = brand_count
    args.msort_count = msort_count

    # else para
    epoch = 3
    train_batch_size = 32
    test_batch_size = 50
    eval_batch_size = 50
    checkpoint_dir = '../save_path/ckpt'

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        # build model
        model = Model(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        if args.is_training:
            print('training_start !!!')
            lr = 1.0
            start_time = time.time()
            # 在训练集上训练epoch轮
            for _ in range(epoch):

                random.shuffle(train_set)

                epoch_size = round(len(train_set) / train_batch_size)
                train_loss_sum = 0.
                train_gauc_sum = 0.
                train_data_cnt = 0.
                for _, uij in DataInput(train_set, train_batch_size):
                    train_loss, train_gauc, current_train_batch_size = model.train(sess, uij, lr, keep_prob=0.95)
                    train_loss_sum += train_loss * current_train_batch_size
                    train_gauc_sum += train_gauc * current_train_batch_size
                    train_data_cnt += current_train_batch_size
                    # 每1000批数据保存一次模型, 并计算1000批数据的平均损失
                    if model.global_step.eval() % 1000 == 0:
                        # model.save(sess, checkpoint_dir)
                        print('Epoch %d Global_step %d\tTrain_loss: %.6f\tTrain_gauc: %.6f' %
                              (model.global_epoch_step.eval(), model.global_step.eval(),
                               train_loss_sum / train_data_cnt, train_gauc_sum / train_data_cnt))
                        sys.stdout.flush()
                        train_loss_sum = 0.
                        train_gauc_sum = 0.
                        train_data_cnt = 0.
                    # 每10000批数据在验证集上验证一次
                    if model.global_step.eval() % 5000 == 0 and args.is_eval:
                        eval_loss_sum = 0.
                        eval_data_cnt = 0.
                        eval_gauc_sum = 0.
                        for _, uij in DataInput(test_set, eval_batch_size):
                            eval_loss, eval_gauc, current_eval_batch_size = model.eval(sess, uij, keep_prob=1.0)
                            eval_loss_sum += eval_loss * current_eval_batch_size
                            eval_gauc_sum += eval_gauc * current_eval_batch_size
                            eval_data_cnt += current_eval_batch_size
                        print('\tEval_loss: %.6f\tEval_gauc: %.6f' %
                              (eval_loss_sum / eval_data_cnt, eval_gauc_sum / eval_data_cnt))
                        sys.stdout.flush()

                    if model.global_step.eval() % 336000 == 0:
                        lr = 0.1

                print('Epoch %d DONE\tCost time: %.2f' %
                      (model.global_epoch_step.eval(), time.time() - start_time))
                sys.stdout.flush()
                model.global_epoch_step_op.eval()
            print('training_finished !!!')
        elif args.is_predict:
            print('predicting_start !!!')
            model.restore(sess, checkpoint_dir)
            out_file_skn = open("../predict/prediction.txt", "w")
            for _, uij in DataInputPredict(test_set, test_batch_size):
                loss, output = model.test(sess, uij, keep_prob=1.0)
                pre_index = np.argsort(-output, axis=1)[:, 0:20]
                for y in range(len(uij[0])):
                    # userID
                    out_file_skn.write(str(uij[0][y]))
                    pre_skn = pre_index[y]
                    # itemID
                    for k in pre_skn:
                        out_file_skn.write("\t%s" % item_key[k])
                    out_file_skn.write("\n")
            print('predicting_finished !!!')
