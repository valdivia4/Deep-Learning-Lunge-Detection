#!/usr/bin/env python
# coding: utf-8


from keras import backend as K
import tensorflow as tf
import datetime
import sys
sys.path.insert(0,'../utils')
from model_configs import get_config
from train import run_training

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config = tf_config)
K.tensorflow_backend._get_available_gpus()

config_name = 'resnet'
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
import pandas as pd 
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 


def build_model(input_shape, nb_classes, config):
	n_feature_maps = 64

	input_layer = keras.layers.Input(input_shape)
	
	# BLOCK 1 

	conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
	conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
	conv_x = keras.layers.Activation('relu')(conv_x)

	conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
	conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
	conv_y = keras.layers.Activation('relu')(conv_y)

	conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
	conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

	# expand channels for the sum 
	shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
	shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

	output_block_1 = keras.layers.add([shortcut_y, conv_z])
	output_block_1 = keras.layers.Activation('relu')(output_block_1)

	# BLOCK 2 

	conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
	conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
	conv_x = keras.layers.Activation('relu')(conv_x)

	conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
	conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
	conv_y = keras.layers.Activation('relu')(conv_y)

	conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
	conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

	# expand channels for the sum 
	shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
	shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

	output_block_2 = keras.layers.add([shortcut_y, conv_z])
	output_block_2 = keras.layers.Activation('relu')(output_block_2)

	# BLOCK 3 

	conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
	conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
	conv_x = keras.layers.Activation('relu')(conv_x)

	conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
	conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
	conv_y = keras.layers.Activation('relu')(conv_y)

	conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
	conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

	# no need to expand channels because they are equal 
	shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

	output_block_3 = keras.layers.add([shortcut_y, conv_z])
	output_block_3 = keras.layers.Activation('relu')(output_block_3)

	# FINAL 
	
	gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

	output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer)

	model.compile(loss=config.loss, optimizer=config.optimizer,
		metrics=config.metrics)
	K.set_value(model.optimizer.lr, config.learning_rate)

	return model

path_to_trainset = '../training_windows/label_model_windows/'
X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple)) 

#output_directory = create_directory(output_directory)
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

