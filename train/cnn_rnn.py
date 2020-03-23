
#!/usr/bin/env python
# coding: utf-8


from keras import backend as K
import tensorflow as tf
import sys
sys.path.insert(0,'../utils')
from model_configs import get_config
from train import run_training

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config = tf_config)
K.tensorflow_backend._get_available_gpus()

config_name = 'cnn_rnn_search'
config = get_config(config_name)


# From Github repo: https://github.com/hfawaz/dl-4-tsc
"""
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                = {Data Mining and Knowledge Discovery},
  Year                     = {2019}
}
"""


# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np

import matplotlib
matplotlib.use('agg')

from keras.layers import Lambda



def build_model(input_shape, nb_classes, config):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=16, strides=4, padding='valid')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='valid')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv_output = conv2
    rnn_input1 = Lambda(lambda x: x[:, 0:32, :])(conv_output)
    rnn_input2 = Lambda(lambda x: x[:, -1:14:-1, :])(conv_output)
    rnn_model1 = keras.layers.LSTM(100)
    rnn_model2 = keras.layers.LSTM(100)
    rnn_layer1 = rnn_model1(rnn_input1)
    rnn_layer2 = rnn_model2(rnn_input2)
    rnn_layer = keras.layers.concatenate([rnn_layer1, rnn_layer2])
    dropout = keras.layers.Dropout(0.5)(rnn_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(dropout)


    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=config.loss, optimizer=config.optimizer,
        metrics=config.metrics)

    K.set_value(model.optimizer.lr, config.learning_rate)

    return model 

path_to_trainset = '../training_windows/label_model_windows/'
X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple)) 

input_shape = X_trainval.shape[1:]
nb_classes = 2

num_trainings = 20
for __ in range(num_trainings):
	config = get_config(config_name)
	model = build_model(input_shape, nb_classes, config)
	model.summary()

	run_training(model, config)
	if not config.hyper_search:
		break

