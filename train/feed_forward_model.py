#
#/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import os
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import datetime
import sys
sys.path.insert(0,'../utils')
from utils import get_model_metrics
from model_configs import get_config
from train import run_training

config_name = 'feed_forward'
config = get_config(config_name)

path_to_trainset = '../training_windows/label_model_windows/'
X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple)) 
__, w, f = X_trainval.shape
input_dim = w*f

def build_model(config):
    print(config.hidden_layers, config.l2_reg) 
    model = Sequential()
    if len(config.hidden_layers) > 0:
        model.add(Dense(config.hidden_layers[0], input_dim=input_dim, kernel_regularizer=l2(config.l2_reg)))
        if config.batch_norm:
            model.add(BatchNormalization())
        model.add(Activation(config.activation))
        for n in config.hidden_layers[1:]:
            model.add(Dense(n))
            if config.batch_norm:
                model.add(BatchNormalization())
            model.add(Activation(config.activation))
        model.add(Dense(1))
    else:
        model.add(Dense(1, input_dim = input_dim, kernel_regularizer=l2(config.l2_reg)))
    model.add(Activation(config.output_activation))

    model.compile(optimizer=config.optimizer,
                  loss=config.loss,
                  metrics=config.metrics)
    K.set_value(model.optimizer.lr, config.learning_rate)

    return model

num_trainings = 20 #used for model hyperparameter searching
for __ in range(num_trainings):
    config = get_config(config_name) #randomly initialized config for hyperparameter search
    model = build_model(config)
    model.summary()
    run_training(model, config)
    if not config.hyper_search:
        break

