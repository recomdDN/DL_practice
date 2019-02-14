mkdir dnn_save_path
mkdir dnn_best_model
source activate tensorflow-1.4
#CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train.py train DNN  >train_dein2.log 2>&1 &
python ./script/train.py train DNN > train_dnn.log 2>&1 &