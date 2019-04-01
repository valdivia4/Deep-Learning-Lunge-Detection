#
#/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import datetime
import sys
sys.path.insert(0,'../utils')
from utils import get_model_metrics
from configs import get_config
from train import run_training

config=get_config('feed_forward')

path_to_trainset = '../data/label_model_windows/'
X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple)) 
__, w, f = X_trainval.shape
input_dim = w*f

model = Sequential()
if len(config.hidden_layers) > 0:
    model.add(Dense(config.hidden_layers[0], input_dim=input_dim))
    model.add(Activation(config.activation))
    if config.batch_norm:
        model.add(BatchNormalization())
    for n in config.hidden_layers[1:]:
        model.add(Dense(n))
        model.add(Activation(config.activation))
        if config.batch_norm:
            model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation(config.output_activation))



model.compile(optimizer=config.optimizer,
              loss=config.loss,
              metrics=config.metrics)

run_training(model, config)

